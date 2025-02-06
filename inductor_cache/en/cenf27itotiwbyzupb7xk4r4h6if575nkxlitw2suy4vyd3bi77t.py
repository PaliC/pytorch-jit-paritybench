# AOT ID: ['68_forward']
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


# kernel path: inductor_cache/kz/ckzpwvx3f2c5vljnru4jd2tpszg6wg74u42w3lsvp2neo6cby2ph.py
# Topologically Sorted Source Nodes: [conv2d, batch_norm, xout], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   conv2d => convolution
#   xout => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 3)
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


# kernel path: inductor_cache/tk/ctk4zy4luneumbbxsr5f3t4rysb6hxgmnw2uvki2wq2zu27lknij.py
# Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, xout_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_1
#   conv2d_1 => convolution_1
#   xout_1 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_8, %primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 12)
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


# kernel path: inductor_cache/co/ccouezy4im2dyutuiim36skd5ng73flxr24q3d77yi2baswihj5x.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_2 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
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


# kernel path: inductor_cache/yy/cyyvnwqmgkch3xthounjn3namn245o6qapdzvk4pwqghactz7vgk.py
# Topologically Sorted Source Nodes: [conv2d_2, batch_norm_2, xout_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_5, mul_7, mul_8, sub_2
#   conv2d_2 => convolution_2
#   xout_2 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_14, %primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 12)
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


# kernel path: inductor_cache/3y/c3y5b7oer6q363bb4stukmc5vxriz7bu7cos7wosgucgj2wijioj.py
# Topologically Sorted Source Nodes: [hx_1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_1 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_4 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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


# kernel path: inductor_cache/p4/cp44d4irpjwdasfxao2cqxsznbgqgrwkt2e72otge3f2i2udbrcc.py
# Topologically Sorted Source Nodes: [conv2d_3, batch_norm_3, xout_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, mul_10, mul_11, sub_3
#   conv2d_3 => convolution_3
#   xout_3 => relu_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_20, %primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 12)
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


# kernel path: inductor_cache/6k/c6kug2d3c3i5ytc6utdi7adrwpv23mms54futyvikretos7kzcot.py
# Topologically Sorted Source Nodes: [hx_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_2 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_6 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ym/cymdg7hy755b3ohek7435g2h3hydn3spv5jl57luyvuncerqxusy.py
# Topologically Sorted Source Nodes: [conv2d_4, batch_norm_4, xout_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_4 => add_9, mul_13, mul_14, sub_4
#   conv2d_4 => convolution_4
#   xout_4 => relu_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_26, %primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 12)
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


# kernel path: inductor_cache/op/cop33rgm4wuf5ykdgnikajskq2lvwd3fk7r5gfgpi63ij74ac3n4.py
# Topologically Sorted Source Nodes: [hx_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hx_3 => getitem_6, getitem_7
# Graph fragment:
#   %getitem_6 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 0), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_8 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2j/c2jvjswgvnibmv4xqjy5l2fg4xgkybpw2xbbfiqp56hvdotc35p2.py
# Topologically Sorted Source Nodes: [conv2d_5, batch_norm_5, xout_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_5 => add_11, mul_16, mul_17, sub_5
#   conv2d_5 => convolution_5
#   xout_5 => relu_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 12)
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


# kernel path: inductor_cache/lf/clf6v5jbgoxjn42wyzpgkvupb74zlri2rlevyq6ljw2nhlllufwu.py
# Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_6 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_38, %primals_39, [1, 1], [2, 2], [2, 2], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 12)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ct/cctklrhshyqtcby2pq6cuiw4bf6l6sc2aelze73q7mpck3b3vzq7.py
# Topologically Sorted Source Nodes: [hx_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_4 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_6, %relu_5], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 24)
    x0 = (xindex % 16)
    x2 = xindex // 384
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 192*x2), tmp4 & xmask, other=0.0)
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
    tmp26 = tl.full([1], 24, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (x0 + 16*((-12) + x1) + 192*x2), tmp25 & xmask, other=0.0)
    tmp29 = tl.where(tmp4, tmp24, tmp28)
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/im/cimayjaganptetzed5ftyqlouw3hi4gttca7ax767vdwlsyoc4nt.py
# Topologically Sorted Source Nodes: [src], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   src => convert_element_type_17
# Graph fragment:
#   %convert_element_type_17 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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


# kernel path: inductor_cache/yo/cyo3jcotmzva2w4mm26773dd2mrsimgx6knzvr3xn5ur33cwxcvt.py
# Topologically Sorted Source Nodes: [src], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   src => add_17, clamp_max
# Graph fragment:
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_17, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_17, 3), kwargs = {})
triton_poi_fused_add_clamp_13 = async_compile.triton('triton_poi_fused_add_clamp_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i3/ci3de25vjvcdqfig2mpoa4lnchkqetzy5dkycc5oua753aijq4qq.py
# Topologically Sorted Source Nodes: [src], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   src => add_16, clamp_max_2, clamp_min, clamp_min_2, convert_element_type_16, iota, mul_24, sub_10, sub_8
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_16, 0.5), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, 0.5), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_24, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_8, 0.0), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_19), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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


# kernel path: inductor_cache/2t/c2tprdkbohnq2wpbnztlzvs34cvian6q4xwmb6z7mpe256ymafog.py
# Topologically Sorted Source Nodes: [src], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   src => _unsafe_index, _unsafe_index_1, add_20, mul_26, sub_11
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_7, [None, None, %convert_element_type_17, %convert_element_type_19]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_7, [None, None, %convert_element_type_17, %clamp_max_1]), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %clamp_max_2), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_26), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_15 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4o/c4onsmqxexb42rjmqt4gjbj3u5cmcbzdcucbhfgzoqqgmqdhfvuc.py
# Topologically Sorted Source Nodes: [hx_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_5 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_22, %relu_4], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 64) % 24)
    x3 = xindex // 1536
    x4 = (xindex % 64)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x2) + 768*x3), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 4*tmp10 + 16*(x2) + 192*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 4*tmp10 + 16*(x2) + 192*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 24, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 64*((-12) + x2) + 768*x3), tmp31 & xmask, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7z/c7zf5w37omaroykjw2k6harmrzpppxr27w73gj7xb26scgil47uq.py
# Topologically Sorted Source Nodes: [src_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   src_1 => convert_element_type_23
# Graph fragment:
#   %convert_element_type_23 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_17 = async_compile.triton('triton_poi_fused__to_copy_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: inductor_cache/nl/cnlqs6cxaidjwye4llgokqe4uz6vjv6ieiaj7gugusghx3jslld5.py
# Topologically Sorted Source Nodes: [src_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   src_1 => add_26, clamp_max_4
# Graph fragment:
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_23, 1), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_26, 7), kwargs = {})
triton_poi_fused_add_clamp_18 = async_compile.triton('triton_poi_fused_add_clamp_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_18(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp11 = tl.full([1], 7, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kl/cklg3zgyiupipydkkeattrjgw2vzewra23jqkx47aive56fz7vjs.py
# Topologically Sorted Source Nodes: [src_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   src_1 => add_25, clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_22, iota_2, mul_32, sub_16, sub_18
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_22, 0.5), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, 0.5), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_32, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_16, 0.0), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_25), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_18, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: inductor_cache/sh/cshoxldwyoxioiwsanwg6zyoxo7fvi7kbzltdext3d5nztgmuxz2.py
# Topologically Sorted Source Nodes: [src_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   src_1 => _unsafe_index_4, _unsafe_index_5, add_29, mul_34, sub_19
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_8, [None, None, %convert_element_type_23, %convert_element_type_25]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_8, [None, None, %convert_element_type_23, %clamp_max_5]), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %clamp_max_6), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_34), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_20 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/vg/cvgcrq7zufa6rjd4z2eegt3ehz7vkxdecydmvtzm6dxhkuargde4.py
# Topologically Sorted Source Nodes: [hx_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_6 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_31, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 24)
    x3 = xindex // 6144
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 3072*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 8, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 8*tmp10 + 64*(x2) + 768*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 8*tmp10 + 64*(x2) + 768*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 24, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 256*((-12) + x2) + 3072*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/h7/ch7ofhs7ooldqpadhgztgvj6d5ztmy344eoavrdey6rzznl57fre.py
# Topologically Sorted Source Nodes: [src_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   src_2 => convert_element_type_29
# Graph fragment:
#   %convert_element_type_29 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_22 = async_compile.triton('triton_poi_fused__to_copy_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_22(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: inductor_cache/jy/cjydmoau74tn57fpeq2wgivum63ky4bfh7r7ilg6zcmqou4pkfef.py
# Topologically Sorted Source Nodes: [src_2], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   src_2 => add_35, clamp_max_8
# Graph fragment:
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_29, 1), kwargs = {})
#   %clamp_max_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_35, 15), kwargs = {})
triton_poi_fused_add_clamp_23 = async_compile.triton('triton_poi_fused_add_clamp_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tmp11 = tl.full([1], 15, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uo/cuobqxg5u3ptowtc67cd26am3a5mspnarsuwtkw52l5ss5jfarer.py
# Topologically Sorted Source Nodes: [src_2], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   src_2 => add_34, clamp_max_10, clamp_min_10, clamp_min_8, convert_element_type_28, iota_4, mul_40, sub_24, sub_26
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_28 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_28, 0.5), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_40, 0.5), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_31), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_26, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_24 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_24', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_24(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: inductor_cache/ir/circ3cygo2unawdcqemlu7s7v4mpifjpxbqdvavvmf5gee67htpy.py
# Topologically Sorted Source Nodes: [src_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   src_2 => _unsafe_index_8, _unsafe_index_9, add_38, mul_42, sub_27
# Graph fragment:
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %convert_element_type_29, %convert_element_type_31]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %convert_element_type_29, %clamp_max_9]), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %clamp_max_10), kwargs = {})
#   %add_38 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_42), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_25 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/gv/cgvxk3pouozebo4x6srfghvznry7r4qshtsxzm4htbxcq266wlq2.py
# Topologically Sorted Source Nodes: [hx_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_7 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_40, %relu_2], 1), kwargs = {})
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 24)
    x3 = xindex // 24576
    x4 = (xindex % 1024)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 1024*(x2) + 12288*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 16, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 16*tmp10 + 256*(x2) + 3072*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 16*tmp10 + 256*(x2) + 3072*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 24, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 1024*((-12) + x2) + 12288*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cobg4ren2k7kka2sje6vts24birvwy5j3dktrnpqw35vwkx3ymhl.py
# Topologically Sorted Source Nodes: [src_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   src_3 => convert_element_type_35
# Graph fragment:
#   %convert_element_type_35 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
triton_poi_fused__to_copy_27 = async_compile.triton('triton_poi_fused__to_copy_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_27(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ka/ckaq6vaijhlnuobomec7doyk3c2ekqritl6zvsvglbrwavsrc2ur.py
# Topologically Sorted Source Nodes: [src_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   src_3 => add_44, clamp_max_12
# Graph fragment:
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_35, 1), kwargs = {})
#   %clamp_max_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_44, 31), kwargs = {})
triton_poi_fused_add_clamp_28 = async_compile.triton('triton_poi_fused_add_clamp_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_28(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/vj/cvjmlguwhmbkf72i6j34wb5x55skxveojp5tanuo26fez5gootbb.py
# Topologically Sorted Source Nodes: [src_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   src_3 => add_43, clamp_max_14, clamp_min_12, clamp_min_14, convert_element_type_34, iota_6, mul_48, sub_32, sub_34
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_6, torch.float32), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_34, 0.5), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, 0.5), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_48, 0.5), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_32, 0.0), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_37), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_34, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_29 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_29(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ti/ctigwlsl6n7bkwuufgtdiegznbs463gmisec6hq3mvgxt4z4yagj.py
# Topologically Sorted Source Nodes: [src_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   src_3 => _unsafe_index_12, _unsafe_index_13, add_47, mul_50, sub_35
# Graph fragment:
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_10, [None, None, %convert_element_type_35, %convert_element_type_37]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_10, [None, None, %convert_element_type_35, %clamp_max_13]), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %clamp_max_14), kwargs = {})
#   %add_47 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_50), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_30 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/zl/czl5l7xngs7q7jn42nddzxrfje3j6r4cd7ngwa4ktqlyju773ibg.py
# Topologically Sorted Source Nodes: [hx_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hx_8 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_49, %relu_1], 1), kwargs = {})
triton_poi_fused_cat_31 = async_compile.triton('triton_poi_fused_cat_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 24)
    x3 = xindex // 98304
    x4 = (xindex % 4096)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 4096*(x2) + 49152*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 32, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 32*tmp10 + 1024*(x2) + 12288*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 32*tmp10 + 1024*(x2) + 12288*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 24, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 4096*((-12) + x2) + 49152*x3), tmp31, other=0.0)
    tmp35 = tl.where(tmp4, tmp30, tmp34)
    tl.store(out_ptr0 + (x5), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/js/cjssp7npjlxgu7v5jcosxjo7dwrxt2adrqeis6kmyohvlkpfthgr.py
# Topologically Sorted Source Nodes: [conv2d_11, batch_norm_11, xout_11, add], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add => add_52
#   batch_norm_11 => add_51, mul_54, mul_55, sub_39
#   conv2d_11 => convolution_11
#   xout_11 => relu_11
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_4, %primals_68, %primals_69, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_91), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %unsqueeze_93), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %unsqueeze_95), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_51,), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_11, %relu), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 3)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), None)
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
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (3, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (3, ), (1, ))
    assert_size_stride(primals_4, (3, ), (1, ))
    assert_size_stride(primals_5, (3, ), (1, ))
    assert_size_stride(primals_6, (3, ), (1, ))
    assert_size_stride(primals_7, (3, ), (1, ))
    assert_size_stride(primals_8, (12, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_9, (12, ), (1, ))
    assert_size_stride(primals_10, (12, ), (1, ))
    assert_size_stride(primals_11, (12, ), (1, ))
    assert_size_stride(primals_12, (12, ), (1, ))
    assert_size_stride(primals_13, (12, ), (1, ))
    assert_size_stride(primals_14, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_15, (12, ), (1, ))
    assert_size_stride(primals_16, (12, ), (1, ))
    assert_size_stride(primals_17, (12, ), (1, ))
    assert_size_stride(primals_18, (12, ), (1, ))
    assert_size_stride(primals_19, (12, ), (1, ))
    assert_size_stride(primals_20, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_21, (12, ), (1, ))
    assert_size_stride(primals_22, (12, ), (1, ))
    assert_size_stride(primals_23, (12, ), (1, ))
    assert_size_stride(primals_24, (12, ), (1, ))
    assert_size_stride(primals_25, (12, ), (1, ))
    assert_size_stride(primals_26, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_27, (12, ), (1, ))
    assert_size_stride(primals_28, (12, ), (1, ))
    assert_size_stride(primals_29, (12, ), (1, ))
    assert_size_stride(primals_30, (12, ), (1, ))
    assert_size_stride(primals_31, (12, ), (1, ))
    assert_size_stride(primals_32, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_33, (12, ), (1, ))
    assert_size_stride(primals_34, (12, ), (1, ))
    assert_size_stride(primals_35, (12, ), (1, ))
    assert_size_stride(primals_36, (12, ), (1, ))
    assert_size_stride(primals_37, (12, ), (1, ))
    assert_size_stride(primals_38, (12, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_39, (12, ), (1, ))
    assert_size_stride(primals_40, (12, ), (1, ))
    assert_size_stride(primals_41, (12, ), (1, ))
    assert_size_stride(primals_42, (12, ), (1, ))
    assert_size_stride(primals_43, (12, ), (1, ))
    assert_size_stride(primals_44, (12, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_45, (12, ), (1, ))
    assert_size_stride(primals_46, (12, ), (1, ))
    assert_size_stride(primals_47, (12, ), (1, ))
    assert_size_stride(primals_48, (12, ), (1, ))
    assert_size_stride(primals_49, (12, ), (1, ))
    assert_size_stride(primals_50, (12, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_51, (12, ), (1, ))
    assert_size_stride(primals_52, (12, ), (1, ))
    assert_size_stride(primals_53, (12, ), (1, ))
    assert_size_stride(primals_54, (12, ), (1, ))
    assert_size_stride(primals_55, (12, ), (1, ))
    assert_size_stride(primals_56, (12, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_57, (12, ), (1, ))
    assert_size_stride(primals_58, (12, ), (1, ))
    assert_size_stride(primals_59, (12, ), (1, ))
    assert_size_stride(primals_60, (12, ), (1, ))
    assert_size_stride(primals_61, (12, ), (1, ))
    assert_size_stride(primals_62, (12, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_63, (12, ), (1, ))
    assert_size_stride(primals_64, (12, ), (1, ))
    assert_size_stride(primals_65, (12, ), (1, ))
    assert_size_stride(primals_66, (12, ), (1, ))
    assert_size_stride(primals_67, (12, ), (1, ))
    assert_size_stride(primals_68, (3, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_69, (3, ), (1, ))
    assert_size_stride(primals_70, (3, ), (1, ))
    assert_size_stride(primals_71, (3, ), (1, ))
    assert_size_stride(primals_72, (3, ), (1, ))
    assert_size_stride(primals_73, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 3, 64, 64), (12288, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 3, 64, 64), (12288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d, batch_norm, xout], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_3, primals_4, primals_5, primals_6, primals_7, buf2, 49152, grid=grid(49152), stream=stream0)
        del primals_3
        del primals_7
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 12, 64, 64), (49152, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_1, batch_norm_1, xout_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf4, primals_9, primals_10, primals_11, primals_12, primals_13, buf5, 196608, grid=grid(196608), stream=stream0)
        del primals_13
        del primals_9
        buf6 = empty_strided_cuda((4, 12, 32, 32), (12288, 1024, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 12, 32, 32), (12288, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_2.run(buf5, buf6, buf7, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf6, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 12, 32, 32), (12288, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_2, batch_norm_2, xout_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf9, primals_15, primals_16, primals_17, primals_18, primals_19, buf10, 49152, grid=grid(49152), stream=stream0)
        del primals_15
        del primals_19
        buf11 = empty_strided_cuda((4, 12, 16, 16), (3072, 256, 16, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 12, 16, 16), (3072, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_4.run(buf10, buf11, buf12, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf11, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((4, 12, 16, 16), (3072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_3, batch_norm_3, xout_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf14, primals_21, primals_22, primals_23, primals_24, primals_25, buf15, 12288, grid=grid(12288), stream=stream0)
        del primals_21
        del primals_25
        buf16 = empty_strided_cuda((4, 12, 8, 8), (768, 64, 8, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 12, 8, 8), (768, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_6.run(buf15, buf16, buf17, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf16, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 12, 8, 8), (768, 64, 8, 1))
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided_cuda((4, 12, 8, 8), (768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_4, batch_norm_4, xout_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf19, primals_27, primals_28, primals_29, primals_30, primals_31, buf20, 3072, grid=grid(3072), stream=stream0)
        del primals_27
        del primals_31
        buf21 = empty_strided_cuda((4, 12, 4, 4), (192, 16, 4, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 12, 4, 4), (192, 16, 4, 1), torch.int8)
        # Topologically Sorted Source Nodes: [hx_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf20, buf21, buf22, 768, grid=grid(768), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf21, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 12, 4, 4), (192, 16, 4, 1))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 12, 4, 4), (192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_5, batch_norm_5, xout_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf24, primals_33, primals_34, primals_35, primals_36, primals_37, buf25, 768, grid=grid(768), stream=stream0)
        del primals_33
        del primals_37
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_38, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 12, 4, 4), (192, 16, 4, 1))
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(buf27, primals_39, 768, grid=grid(768), stream=stream0)
        del primals_39
        buf28 = empty_strided_cuda((4, 24, 4, 4), (384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf27, primals_40, primals_41, primals_42, primals_43, buf25, buf28, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 12, 4, 4), (192, 16, 4, 1))
        buf30 = buf29; del buf29  # reuse
        buf31 = empty_strided_cuda((4, 12, 4, 4), (192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_7, batch_norm_7, xout_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf30, primals_45, primals_46, primals_47, primals_48, primals_49, buf31, 768, grid=grid(768), stream=stream0)
        del primals_45
        buf32 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf32, 8, grid=grid(8), stream=stream0)
        buf33 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_13.run(buf33, 8, grid=grid(8), stream=stream0)
        buf34 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf34, 8, grid=grid(8), stream=stream0)
        buf35 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_13.run(buf35, 8, grid=grid(8), stream=stream0)
        buf36 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [src], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14.run(buf36, 8, grid=grid(8), stream=stream0)
        buf37 = empty_strided_cuda((4, 12, 8, 8), (768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_15.run(buf32, buf34, buf31, buf35, buf36, buf37, 3072, grid=grid(3072), stream=stream0)
        buf38 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14.run(buf38, 8, grid=grid(8), stream=stream0)
        buf39 = empty_strided_cuda((4, 24, 8, 8), (1536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf37, buf33, buf34, buf31, buf35, buf36, buf38, buf20, buf39, 6144, grid=grid(6144), stream=stream0)
        del buf31
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 12, 8, 8), (768, 64, 8, 1))
        buf41 = buf40; del buf40  # reuse
        buf42 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [conv2d_8, batch_norm_8, xout_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf41, primals_51, primals_52, primals_53, primals_54, primals_55, buf42, 3072, grid=grid(3072), stream=stream0)
        del primals_51
        buf43 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf43, 16, grid=grid(16), stream=stream0)
        buf44 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_18.run(buf44, 16, grid=grid(16), stream=stream0)
        buf45 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf45, 16, grid=grid(16), stream=stream0)
        buf46 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_18.run(buf46, 16, grid=grid(16), stream=stream0)
        buf47 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [src_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19.run(buf47, 16, grid=grid(16), stream=stream0)
        buf48 = empty_strided_cuda((4, 12, 16, 16), (3072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_20.run(buf43, buf45, buf42, buf46, buf47, buf48, 12288, grid=grid(12288), stream=stream0)
        buf49 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19.run(buf49, 16, grid=grid(16), stream=stream0)
        buf50 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf48, buf44, buf45, buf42, buf46, buf47, buf49, buf15, buf50, 24576, grid=grid(24576), stream=stream0)
        del buf42
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf52 = buf51; del buf51  # reuse
        buf53 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [conv2d_9, batch_norm_9, xout_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf52, primals_57, primals_58, primals_59, primals_60, primals_61, buf53, 12288, grid=grid(12288), stream=stream0)
        del primals_57
        buf54 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(buf54, 32, grid=grid(32), stream=stream0)
        buf55 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src_2], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_23.run(buf55, 32, grid=grid(32), stream=stream0)
        buf56 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src_2], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(buf56, 32, grid=grid(32), stream=stream0)
        buf57 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src_2], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_23.run(buf57, 32, grid=grid(32), stream=stream0)
        buf58 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [src_2], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_24.run(buf58, 32, grid=grid(32), stream=stream0)
        buf59 = empty_strided_cuda((4, 12, 32, 32), (12288, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_25.run(buf54, buf56, buf53, buf57, buf58, buf59, 49152, grid=grid(49152), stream=stream0)
        buf60 = empty_strided_cuda((32, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src_2], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_24.run(buf60, 32, grid=grid(32), stream=stream0)
        buf61 = empty_strided_cuda((4, 24, 32, 32), (24576, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_26.run(buf59, buf55, buf56, buf53, buf57, buf58, buf60, buf10, buf61, 98304, grid=grid(98304), stream=stream0)
        del buf53
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf63 = buf62; del buf62  # reuse
        buf64 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [conv2d_10, batch_norm_10, xout_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf63, primals_63, primals_64, primals_65, primals_66, primals_67, buf64, 49152, grid=grid(49152), stream=stream0)
        del primals_63
        buf65 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_27.run(buf65, 64, grid=grid(64), stream=stream0)
        buf66 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [src_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_28.run(buf66, 64, grid=grid(64), stream=stream0)
        buf67 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_27.run(buf67, 64, grid=grid(64), stream=stream0)
        buf68 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [src_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_28.run(buf68, 64, grid=grid(64), stream=stream0)
        buf69 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [src_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_29.run(buf69, 64, grid=grid(64), stream=stream0)
        buf70 = empty_strided_cuda((4, 12, 64, 64), (49152, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_30.run(buf65, buf67, buf64, buf68, buf69, buf70, 196608, grid=grid(196608), stream=stream0)
        buf71 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [src_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_29.run(buf71, 64, grid=grid(64), stream=stream0)
        buf72 = empty_strided_cuda((4, 24, 64, 64), (98304, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_31.run(buf70, buf66, buf67, buf64, buf68, buf69, buf71, buf5, buf72, 393216, grid=grid(393216), stream=stream0)
        del buf70
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 3, 64, 64), (12288, 4096, 64, 1))
        buf74 = buf73; del buf73  # reuse
        buf75 = reinterpret_tensor(buf64, (4, 3, 64, 64), (12288, 4096, 64, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [conv2d_11, batch_norm_11, xout_11, add], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32.run(buf74, primals_69, primals_70, primals_71, primals_72, primals_73, buf2, buf75, 49152, grid=grid(49152), stream=stream0)
        del primals_69
    return (buf75, primals_1, primals_2, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_56, primals_58, primals_59, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_73, buf1, buf2, buf4, buf5, buf6, buf7, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf19, buf20, buf21, buf22, buf24, buf25, buf27, buf28, buf30, buf32, buf33, buf34, buf35, buf36, buf38, buf39, buf41, buf43, buf44, buf45, buf46, buf47, buf49, buf50, buf52, buf54, buf55, buf56, buf57, buf58, buf60, buf61, buf63, buf65, buf66, buf67, buf68, buf69, buf71, buf72, buf74, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((3, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((12, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((12, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((12, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((12, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((12, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((12, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((3, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
