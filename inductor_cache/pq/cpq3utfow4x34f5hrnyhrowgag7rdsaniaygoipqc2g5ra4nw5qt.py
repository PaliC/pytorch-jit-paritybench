# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/s6/cs6c2cac4jxcyrgqfswdxridngn3ny3abpfnpgyf3sxwsnytjk4j.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_1 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_0 = async_compile.triton('triton_poi_fused_reflection_pad2d_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 70)
    x1 = ((xindex // 70) % 70)
    x2 = xindex // 4900
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-3) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-3) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fr/cfrrzrspart3djbef36xdfvomwwxqdl27yojrutmwvz2xfad5ies.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => convolution
#   input_3 => add_1, mul_1, mul_2, sub_4
#   input_4 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
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


# kernel path: inductor_cache/6g/c6gxeg3hrg6zzcue64wx553rqdrfb3zkjk5dysaqhl2anxnswmod.py
# Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => convolution_1
#   input_6 => add_3, mul_4, mul_5, sub_5
#   input_7 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_8, %primals_9, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_11), kwargs = {})
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
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 128)
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


# kernel path: inductor_cache/ay/cayepphex47swiub6n4kl2nkh6lujbre6npgqtc7t372uhwfsyl3.py
# Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => relu_2
#   input_8 => convolution_2
#   input_9 => add_5, mul_7, mul_8, sub_6
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_14, %primals_15, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
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


# kernel path: inductor_cache/dj/cdjlbct52x3pa4z2gh2fwrpdlkcjzkaaiafzyy6f4j3oesuskoqk.py
# Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_11 => convolution_3
#   input_12 => add_7, mul_10, mul_11, sub_7
#   input_13 => relu_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_20, %primals_21, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
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


# kernel path: inductor_cache/cf/ccfrx5xniv6hfx3zngv5m3mbobvatqygbnzztpd2zg3rsihvpact.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_14 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_3, [None, None, %sub_9, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_9]), kwargs = {})
triton_poi_fused_reflection_pad2d_5 = async_compile.triton('triton_poi_fused_reflection_pad2d_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 10)
    x1 = ((xindex // 10) % 10)
    x2 = xindex // 100
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-7) + tl_math.abs((-1) + x0))) + ((-8)*tl_math.abs((-7) + tl_math.abs((-1) + x1))) + 64*x2), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/gz/cgznsotlogczbzhfqbamkrrcngjeicshezfvfalpteye3hmytots.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_15 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %primals_26, %primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_6 = async_compile.triton('triton_poi_fused_convolution_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ft/cftpb3szzqveg3uih5ubrtvxrzrxv7wmdhlkquntsi4zo4bjnoog.py
# Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_16 => add_9, mul_13, mul_14, sub_12
#   input_17 => relu_4
#   input_18 => _unsafe_index_4, _unsafe_index_5
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_4, [None, None, %sub_9, None]), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_4, [None, None, None, %sub_9]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 10)
    x1 = ((xindex // 10) % 10)
    x4 = xindex // 100
    x2 = ((xindex // 100) % 512)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-7) + tl_math.abs((-1) + x0))) + ((-8)*tl_math.abs((-7) + tl_math.abs((-1) + x1))) + 64*x4), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/h4/ch4a4kd7xblj4g5tnun2yeufvter2nsz5ators7gzwzeqwn34yrg.py
# Topologically Sorted Source Nodes: [input_19, input_20, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_19 => convolution_5
#   input_20 => add_11, mul_16, mul_17, sub_17
#   out => add_12
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_5, %primals_32, %primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_3, %add_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 - tmp4
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp10 / tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp5 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp3 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/5l/c5lcispe5afll7y64nkfrvipz2bosyqgwcey26lrl7e5xuruvl2u.py
# Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_83 => convolution_24
# Graph fragment:
#   %convolution_24 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_146, %primals_147, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/66/c66nwyvorbbsdwumt4kznm6epcv2umqyidfj5ard4jqybkjzzmbx.py
# Topologically Sorted Source Nodes: [input_84, input_85, input_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_84 => add_58, mul_73, mul_74, sub_100
#   input_85 => relu_15
#   input_86 => _unsafe_index_38, _unsafe_index_39
# Graph fragment:
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
#   %_unsafe_index_38 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_15, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_39 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_38, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 70)
    x1 = ((xindex // 70) % 70)
    x4 = xindex // 4900
    x2 = ((xindex // 4900) % 64)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-3) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-3) + x1))) + 4096*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l2/cl2t3uoskzrmo4rhucb2ni2jc52mfhzzyyfdjvjdmu2sriplsqhl.py
# Topologically Sorted Source Nodes: [input_87, input_88], Original ATen: [aten.convolution, aten.tanh]
# Source node to ATen node mapping:
#   input_87 => convolution_25
#   input_88 => tanh
# Graph fragment:
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_39, %primals_152, %primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_25,), kwargs = {})
triton_poi_fused_convolution_tanh_11 = async_compile.triton('triton_poi_fused_convolution_tanh_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_tanh_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_tanh_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_2, (64, 4, 7, 7), (196, 49, 7, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_44, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_56, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_62, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, ), (1, ))
    assert_size_stride(primals_68, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_74, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_80, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_112, (512, ), (1, ))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_134, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (256, ), (1, ))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_147, (64, ), (1, ))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (4, 64, 7, 7), (3136, 49, 7, 1))
    assert_size_stride(primals_153, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 70, 70), (19600, 4900, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0.run(primals_1, buf0, 78400, grid=grid(78400), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf2 = buf1; del buf1  # reuse
        buf3 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf2, primals_3, primals_4, primals_5, primals_6, primals_7, buf3, 1048576, grid=grid(1048576), stream=stream0)
        del primals_3
        del primals_7
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf5, primals_9, primals_10, primals_11, primals_12, primals_13, buf6, 524288, grid=grid(524288), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf8 = buf7; del buf7  # reuse
        buf9 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf8, primals_15, primals_16, primals_17, primals_18, primals_19, buf9, 262144, grid=grid(262144), stream=stream0)
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_20, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf11, primals_21, primals_22, primals_23, primals_24, primals_25, buf12, 131072, grid=grid(131072), stream=stream0)
        del primals_21
        buf13 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf12, buf13, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf15, primals_27, 131072, grid=grid(131072), stream=stream0)
        del primals_27
        buf16 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf15, primals_28, primals_29, primals_30, primals_31, buf16, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf18 = buf17; del buf17  # reuse
        buf19 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_19, input_20, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf18, buf19, primals_33, primals_34, primals_35, primals_36, primals_37, 131072, grid=grid(131072), stream=stream0)
        del primals_33
        del primals_37
        buf20 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf19, buf20, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf22, primals_39, 131072, grid=grid(131072), stream=stream0)
        del primals_39
        buf23 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf22, primals_40, primals_41, primals_42, primals_43, buf23, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf25 = buf24; del buf24  # reuse
        buf26 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_26, input_27, out_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf25, buf26, primals_45, primals_46, primals_47, primals_48, primals_49, 131072, grid=grid(131072), stream=stream0)
        del primals_45
        del primals_49
        buf27 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf26, buf27, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf29, primals_51, 131072, grid=grid(131072), stream=stream0)
        del primals_51
        buf30 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_30, input_31, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf29, primals_52, primals_53, primals_54, primals_55, buf30, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf32 = buf31; del buf31  # reuse
        buf33 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf32, buf33, primals_57, primals_58, primals_59, primals_60, primals_61, 131072, grid=grid(131072), stream=stream0)
        del primals_57
        del primals_61
        buf34 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf33, buf34, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf36, primals_63, 131072, grid=grid(131072), stream=stream0)
        del primals_63
        buf37 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37, input_38, input_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf36, primals_64, primals_65, primals_66, primals_67, buf37, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf39 = buf38; del buf38  # reuse
        buf40 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf39, buf40, primals_69, primals_70, primals_71, primals_72, primals_73, 131072, grid=grid(131072), stream=stream0)
        del primals_69
        del primals_73
        buf41 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf40, buf41, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf43, primals_75, 131072, grid=grid(131072), stream=stream0)
        del primals_75
        buf44 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, input_45, input_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf43, primals_76, primals_77, primals_78, primals_79, buf44, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf46 = buf45; del buf45  # reuse
        buf47 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_47, input_48, out_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf46, buf47, primals_81, primals_82, primals_83, primals_84, primals_85, 131072, grid=grid(131072), stream=stream0)
        del primals_81
        del primals_85
        buf48 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf47, buf48, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf50, primals_87, 131072, grid=grid(131072), stream=stream0)
        del primals_87
        buf51 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_51, input_52, input_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf50, primals_88, primals_89, primals_90, primals_91, buf51, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf53 = buf52; del buf52  # reuse
        buf54 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55, out_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf53, buf54, primals_93, primals_94, primals_95, primals_96, primals_97, 131072, grid=grid(131072), stream=stream0)
        del primals_93
        del primals_97
        buf55 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf54, buf55, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf57, primals_99, 131072, grid=grid(131072), stream=stream0)
        del primals_99
        buf58 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_58, input_59, input_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf57, primals_100, primals_101, primals_102, primals_103, buf58, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf60 = buf59; del buf59  # reuse
        buf61 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_61, input_62, out_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf60, buf61, primals_105, primals_106, primals_107, primals_108, primals_109, 131072, grid=grid(131072), stream=stream0)
        del primals_105
        del primals_109
        buf62 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf61, buf62, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf64, primals_111, 131072, grid=grid(131072), stream=stream0)
        del primals_111
        buf65 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_65, input_66, input_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf64, primals_112, primals_113, primals_114, primals_115, buf65, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf67 = buf66; del buf66  # reuse
        buf68 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69, out_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf67, buf68, primals_117, primals_118, primals_119, primals_120, primals_121, 131072, grid=grid(131072), stream=stream0)
        del primals_117
        del primals_121
        buf69 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf68, buf69, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_6.run(buf71, primals_123, 131072, grid=grid(131072), stream=stream0)
        del primals_123
        buf72 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_72, input_73, input_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_7.run(buf71, primals_124, primals_125, primals_126, primals_127, buf72, 204800, grid=grid(204800), stream=stream0)
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf74 = buf73; del buf73  # reuse
        buf75 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_75, input_76, out_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf74, buf75, primals_129, primals_130, primals_131, primals_132, primals_133, 131072, grid=grid(131072), stream=stream0)
        del primals_129
        del primals_133
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_134, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf76, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf77 = buf76; del buf76  # reuse
        buf78 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_77, input_78, input_79], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf77, primals_135, primals_136, primals_137, primals_138, primals_139, buf78, 262144, grid=grid(262144), stream=stream0)
        del primals_135
        del primals_139
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_140, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf79, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf80 = buf79; del buf79  # reuse
        buf81 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, input_81, input_82], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf80, primals_141, primals_142, primals_143, primals_144, primals_145, buf81, 524288, grid=grid(524288), stream=stream0)
        del primals_141
        del primals_145
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_146, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf82, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf83, primals_147, 1048576, grid=grid(1048576), stream=stream0)
        del primals_147
        buf84 = empty_strided_cuda((4, 64, 70, 70), (313600, 4900, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_84, input_85, input_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_relu_10.run(buf83, primals_148, primals_149, primals_150, primals_151, buf84, 1254400, grid=grid(1254400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_87, input_88], Original ATen: [aten.convolution, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_tanh_11.run(buf86, primals_153, 65536, grid=grid(65536), stream=stream0)
        del primals_153
    return (buf86, primals_2, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_50, primals_52, primals_53, primals_54, primals_55, primals_56, primals_58, primals_59, primals_60, primals_62, primals_64, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_74, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_86, primals_88, primals_89, primals_90, primals_91, primals_92, primals_94, primals_95, primals_96, primals_98, primals_100, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_130, primals_131, primals_132, primals_134, primals_136, primals_137, primals_138, primals_140, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf13, buf15, buf16, buf18, buf20, buf22, buf23, buf25, buf27, buf29, buf30, buf32, buf34, buf36, buf37, buf39, buf41, buf43, buf44, buf46, buf48, buf50, buf51, buf53, buf55, buf57, buf58, buf60, buf62, buf64, buf65, buf67, buf69, buf71, buf72, buf74, buf75, buf77, buf78, buf80, buf81, buf83, buf84, buf86, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 4, 7, 7), (196, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((4, 64, 7, 7), (3136, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
