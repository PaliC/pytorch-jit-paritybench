# AOT ID: ['13_forward']
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
# Topologically Sorted Source Nodes: [bn1, relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   bn1 => add_1, mul_1, mul_2, sub
#   relu1 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
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


# kernel path: inductor_cache/l2/cl2yg645rneni3mm6p74ydzi3w4wtaatbjgrw2a5v5ghowjt2bpb.py
# Topologically Sorted Source Nodes: [downsample_conv1_0, downsample_conv1_1, downsample_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   downsample_conv1_0 => convolution_1
#   downsample_conv1_1 => add_3, mul_4, mul_5, sub_1
#   downsample_conv1_2 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [2, 2], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6h/c6hkqvzabxq5lmrwrjvsznoqmbsyuqjls6hsmee3tt4wob3w2eek.py
# Topologically Sorted Source Nodes: [downsample_conv2_3, downsample_conv2_4, downsample_downsample_res_conv_3, downsample_downsample_res_conv_4, add_1, downsample_relu2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_1 => add_14
#   downsample_conv2_3 => convolution_4
#   downsample_conv2_4 => add_9, mul_13, mul_14, sub_4
#   downsample_downsample_res_conv_3 => convolution_6
#   downsample_downsample_res_conv_4 => add_13, mul_19, mul_20, sub_6
#   downsample_relu2 => relu_5
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %primals_25, %primals_26, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %primals_37, %primals_38, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %add_13), kwargs = {})
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp5, None)
    tl.store(in_out_ptr2 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/vb/cvbcqcccvzgsajlgexglzdbx65m5rz5qrt4svh45xdf6vlygnakj.py
# Topologically Sorted Source Nodes: [layer1_0_conv1_3, layer1_0_conv1_4, add_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_19
#   layer1_0_conv1_3 => convolution_8
#   layer1_0_conv1_4 => add_18, mul_25, mul_26, sub_8
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %primals_49, %primals_50, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_19 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %relu_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x3), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvvbctros56bnjedg2wsuromfswcimj6rp6y4sqjn4j25a2aftp.py
# Topologically Sorted Source Nodes: [transition1_0_0_3, transition1_0_1, transition1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   transition1_0_0_3 => convolution_13
#   transition1_0_1 => add_30, mul_40, mul_41, sub_13
#   transition1_0_2 => relu_10
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_79, %primals_80, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_10 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_30,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 16)
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


# kernel path: inductor_cache/fz/cfz3w6crv4tkh6hqmho6kk62hhcjrcimzpwnujliy7mskqsu4pp7.py
# Topologically Sorted Source Nodes: [transition1_1_0_0_0, transition1_1_0_0_1, transition1_1_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   transition1_1_0_0_0 => convolution_14
#   transition1_1_0_0_1 => add_32, mul_43, mul_44, sub_14
#   transition1_1_0_0_2 => relu_11
# Graph fragment:
#   %convolution_14 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_85, %primals_86, [2, 2], [1, 1], [1, 1], False, [0, 0], 64), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_32,), kwargs = {})
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
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
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


# kernel path: inductor_cache/s3/cs3bytq7ptfg7m5pga47bdf4qgcxi647k4butflxcdueihjurv3s.py
# Topologically Sorted Source Nodes: [transition1_1_0_0_3, transition1_1_0_1, transition1_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   transition1_1_0_0_3 => convolution_15
#   transition1_1_0_1 => add_34, mul_46, mul_47, sub_15
#   transition1_1_0_2 => relu_12
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_91, %primals_92, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_12 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_34,), kwargs = {})
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
    x1 = ((xindex // 64) % 32)
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


# kernel path: inductor_cache/33/c33wgpk43uzuooibvwky5wepxcyoxoopsiogvwwoourv5aayscdl.py
# Topologically Sorted Source Nodes: [stage2_0_branches_0_0_conv1_3, stage2_0_branches_0_0_conv1_4, add_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_39
#   stage2_0_branches_0_0_conv1_3 => convolution_17
#   stage2_0_branches_0_0_conv1_4 => add_38, mul_52, mul_53, sub_17
# Graph fragment:
#   %convolution_17 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %primals_103, %primals_104, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %relu_10), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/eb/cebd5yhvemnq3oaksrkwozqhxccxmq2hmgxc7j6kcv3uewbyfgb5.py
# Topologically Sorted Source Nodes: [stage2_0_branches_1_0_conv1_3, stage2_0_branches_1_0_conv1_4, add_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_46
#   stage2_0_branches_1_0_conv1_3 => convolution_20
#   stage2_0_branches_1_0_conv1_4 => add_45, mul_61, mul_62, sub_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_121, %primals_122, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %relu_12), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/zd/czdpeyocthtkmrdwt2s6eonc4btvkb4naubuwarwhbp23azrn7ew.py
# Topologically Sorted Source Nodes: [stage2_0_fuse_layers_0_1_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   stage2_0_fuse_layers_0_1_2 => add_51, add_52, convert_element_type_46, convert_element_type_47, iota, mul_69, mul_70
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_69, 0), kwargs = {})
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_51, torch.float32), kwargs = {})
#   %add_52 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_46, 0.0), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_52, 0.5), kwargs = {})
#   %convert_element_type_47 : [num_users=7] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_70, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_9 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xt/cxtwqmgmwimlf47anqocpnghpwdoc3grohz2jwksgnmn2ltm4cxq.py
# Topologically Sorted Source Nodes: [stage2_0_branches_0_1_0, stage2_0_branches_0_1_1, stage2_0_branches_0_1_2, stage2_0_fuse_layers_0_1_1, stage2_0_fuse_layers_0_1_2, add_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_55
#   stage2_0_branches_0_1_0 => convolution_18
#   stage2_0_branches_0_1_1 => add_41, mul_55, mul_56, sub_18
#   stage2_0_branches_0_1_2 => relu_14
#   stage2_0_fuse_layers_0_1_1 => add_50, mul_67, mul_68, sub_22
#   stage2_0_fuse_layers_0_1_2 => _unsafe_index
# Graph fragment:
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_39, %primals_109, %primals_110, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_14 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_50, [None, None, %unsqueeze_184, %convert_element_type_47]), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_14, %_unsafe_index), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 16)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
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
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr6 + (tmp28 + 8*tmp24 + 64*x6), None, eviction_policy='evict_last')
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp6
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp9 / tmp34
    tmp36 = tmp35 * tmp11
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp19 + tmp41
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(out_ptr0 + (x5), tmp19, None)
    tl.store(out_ptr1 + (x5), tmp42, None)
''', device_str='cuda')


# kernel path: inductor_cache/cy/ccyzfjrtg2gchinpgoqrlwygeampio5omm772jvesu6yikp5hjv5.py
# Topologically Sorted Source Nodes: [stage2_0_fuse_layers_1_0_0_0_0, stage2_0_fuse_layers_1_0_0_0_1, stage2_0_fuse_layers_1_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   stage2_0_fuse_layers_1_0_0_0_0 => convolution_24
#   stage2_0_fuse_layers_1_0_0_0_1 => add_59, mul_77, mul_78, sub_24
#   stage2_0_fuse_layers_1_0_0_0_2 => relu_18
# Graph fragment:
#   %convolution_24 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_144, %primals_145, [2, 2], [1, 1], [1, 1], False, [0, 0], 16), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_194), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_196), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_77, %unsqueeze_198), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, %unsqueeze_200), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_59,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 16)
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


# kernel path: inductor_cache/5y/c5yo2xuudruy6nuey6uju4zvg2rw4bcglqbgycals3wo7tzpuzzx.py
# Topologically Sorted Source Nodes: [transition2_2_0_0_0, transition2_2_0_0_1, transition2_2_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   transition2_2_0_0_0 => convolution_27
#   transition2_2_0_0_1 => add_66, mul_86, mul_87, sub_27
#   transition2_2_0_0_2 => relu_20
# Graph fragment:
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %primals_162, %primals_163, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_218), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_220), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, %unsqueeze_222), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %unsqueeze_224), kwargs = {})
#   %relu_20 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_66,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 32)
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


# kernel path: inductor_cache/rm/crm3sjoshs4d7t3q555qdvt7xsu4cggplbbnqmmfepovcpxe5zpb.py
# Topologically Sorted Source Nodes: [transition2_2_0_0_3, transition2_2_0_1, transition2_2_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   transition2_2_0_0_3 => convolution_28
#   transition2_2_0_1 => add_68, mul_89, mul_90, sub_28
#   transition2_2_0_2 => relu_21
# Graph fragment:
#   %convolution_28 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %primals_168, %primals_169, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_226), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_228), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %unsqueeze_230), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %unsqueeze_232), kwargs = {})
#   %relu_21 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_68,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 56)
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


# kernel path: inductor_cache/4x/c4x3hudwrb3c76bkqsm42uvspf6bjskajjei5cvrtszff76ie7jp.py
# Topologically Sorted Source Nodes: [stage3_0_branches_2_0_conv1_3, stage3_0_branches_2_0_conv1_4, add_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_10 => add_87
#   stage3_0_branches_2_0_conv1_3 => convolution_36
#   stage3_0_branches_2_0_conv1_4 => add_86, mul_113, mul_114, sub_36
# Graph fragment:
#   %convolution_36 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %primals_216, %primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_290), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_292), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_294), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_296), kwargs = {})
#   %add_87 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %relu_21), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 56)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), xmask)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jq/cjqwpz7xehorggurqapnugpautgtslbp2b22mu57tmnp2alxudwu.py
# Topologically Sorted Source Nodes: [stage2_0_fuse_layers_0_1_2, stage3_0_fuse_layers_0_2_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   stage2_0_fuse_layers_0_1_2 => add_51, add_52, convert_element_type_46, iota, mul_69
#   stage3_0_fuse_layers_0_2_2 => convert_element_type_89, mul_129
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_69, 0), kwargs = {})
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_51, torch.float32), kwargs = {})
#   %add_52 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_46, 0.0), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_52, 0.25), kwargs = {})
#   %convert_element_type_89 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_129, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_15 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.25
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3r/c3rczbs3bcd6sjijajphyczp5mz6guckwyvs7qyzqjorlwc2pjes.py
# Topologically Sorted Source Nodes: [stage3_0_branches_0_1_0, stage3_0_branches_0_1_1, stage3_0_branches_0_1_2, stage3_0_fuse_layers_0_1_1, stage3_0_fuse_layers_0_1_2, add_11, stage3_0_fuse_layers_0_2_1, stage3_0_fuse_layers_0_2_2, add_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
# Source node to ATen node mapping:
#   add_11 => add_96
#   add_12 => add_103
#   stage3_0_branches_0_1_0 => convolution_31
#   stage3_0_branches_0_1_1 => add_75, mul_98, mul_99, sub_31
#   stage3_0_branches_0_1_2 => relu_23
#   stage3_0_fuse_layers_0_1_1 => add_91, mul_119, mul_120, sub_38
#   stage3_0_fuse_layers_0_1_2 => _unsafe_index_1
#   stage3_0_fuse_layers_0_2_1 => add_98, mul_126, mul_127, sub_39
#   stage3_0_fuse_layers_0_2_2 => _unsafe_index_2
# Graph fragment:
#   %convolution_31 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_73, %primals_186, %primals_187, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_250), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_252), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %unsqueeze_254), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_99, %unsqueeze_256), kwargs = {})
#   %relu_23 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_75,), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_306), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_308), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %unsqueeze_310), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %unsqueeze_312), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_91, [None, None, %unsqueeze_184, %convert_element_type_47]), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_23, %_unsafe_index_1), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_315), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_317), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_126, %unsqueeze_319), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, %unsqueeze_321), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_98, [None, None, %unsqueeze_322, %convert_element_type_89]), kwargs = {})
#   %add_103 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_96, %_unsafe_index_2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 16)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x3), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
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
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr6 + (tmp28 + 8*tmp24 + 64*x6), None, eviction_policy='evict_last')
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp6
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp9 / tmp34
    tmp36 = tmp35 * tmp11
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp19 + tmp41
    tmp44 = tl.full([XBLOCK], 4, tl.int32)
    tmp45 = tmp43 + tmp44
    tmp46 = tmp43 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp43)
    tmp49 = tmp48 + tmp44
    tmp50 = tmp48 < 0
    tmp51 = tl.where(tmp50, tmp49, tmp48)
    tmp52 = tl.load(in_ptr12 + (tmp51 + 4*tmp47 + 16*x6), None, eviction_policy='evict_last')
    tmp54 = tmp52 - tmp53
    tmp56 = tmp55 + tmp6
    tmp57 = libdevice.sqrt(tmp56)
    tmp58 = tmp9 / tmp57
    tmp59 = tmp58 * tmp11
    tmp60 = tmp54 * tmp59
    tmp62 = tmp60 * tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp42 + tmp64
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(out_ptr0 + (x5), tmp19, None)
    tl.store(out_ptr1 + (x5), tmp65, None)
''', device_str='cuda')


# kernel path: inductor_cache/yp/cypgepof6xpowj6esseldjlp3npkelysaqdnefoxu4rqm2nx3pug.py
# Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_2_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   stage3_0_fuse_layers_1_2_2 => add_113, add_114, convert_element_type_100, convert_element_type_101, iota_6, mul_144, mul_145
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, 0), kwargs = {})
#   %convert_element_type_100 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_113, torch.float32), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_100, 0.0), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, 0.5), kwargs = {})
#   %convert_element_type_101 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_145, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_17 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/fq/cfqhphcfls2spwy27x3avaryzzk3eyggh6ufcokxlgbr3zmrgkkn.py
# Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_0_0_0_3, stage3_0_fuse_layers_1_0_0_1, add_13, stage3_0_fuse_layers_1_2_1, stage3_0_fuse_layers_1_2_2, add_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_13 => add_110
#   add_14 => add_117
#   stage3_0_fuse_layers_1_0_0_0_3 => convolution_42
#   stage3_0_fuse_layers_1_0_0_1 => add_109, mul_139, mul_140, sub_42
#   stage3_0_fuse_layers_1_2_1 => add_112, mul_142, mul_143, sub_43
#   stage3_0_fuse_layers_1_2_2 => _unsafe_index_3
# Graph fragment:
#   %convolution_42 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %primals_250, %primals_251, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_340), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_342), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_344), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_346), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, %relu_25), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_348), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_350), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, %unsqueeze_352), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, %unsqueeze_354), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_112, [None, None, %unsqueeze_355, %convert_element_type_101]), kwargs = {})
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_110, %_unsafe_index_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 64) % 32)
    x4 = ((xindex // 8) % 8)
    x3 = (xindex % 8)
    x6 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x5), None)
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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
    tmp19 = tmp17 + tmp18
    tmp21 = tl.full([XBLOCK], 4, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr7 + (tmp28 + 4*tmp24 + 16*x6), None, eviction_policy='evict_last')
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp6
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp9 / tmp34
    tmp36 = tmp35 * tmp11
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp19 + tmp41
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(out_ptr0 + (x5), tmp42, None)
''', device_str='cuda')


# kernel path: inductor_cache/2a/c2ao4ea3fhukorheghtw2cse4n5f3i6hvblpckisal47xev2kdde.py
# Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_1_0_0, stage3_0_fuse_layers_2_0_1_0_1, stage3_0_fuse_layers_2_0_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   stage3_0_fuse_layers_2_0_1_0_0 => convolution_47
#   stage3_0_fuse_layers_2_0_1_0_1 => add_125, mul_158, mul_159, sub_47
#   stage3_0_fuse_layers_2_0_1_0_2 => relu_33
# Graph fragment:
#   %convolution_47 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_32, %primals_279, %primals_280, [2, 2], [1, 1], [1, 1], False, [0, 0], 16), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_47, %unsqueeze_381), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_383), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %unsqueeze_385), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %unsqueeze_387), kwargs = {})
#   %relu_33 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_125,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 16)
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


# kernel path: inductor_cache/xa/cxaemctwaie6ztduzmqmceihf3r6ebi4pzrr4gop6aeckut4zfdt.py
# Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_1_0_3, stage3_0_fuse_layers_2_0_1_1, stage3_0_fuse_layers_2_1_0_0_3, stage3_0_fuse_layers_2_1_0_1, add_15, add_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_15 => add_132
#   add_16 => add_133
#   stage3_0_fuse_layers_2_0_1_0_3 => convolution_48
#   stage3_0_fuse_layers_2_0_1_1 => add_127, mul_161, mul_162, sub_48
#   stage3_0_fuse_layers_2_1_0_0_3 => convolution_50
#   stage3_0_fuse_layers_2_1_0_1 => add_131, mul_167, mul_168, sub_50
# Graph fragment:
#   %convolution_48 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_33, %primals_285, %primals_286, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %unsqueeze_389), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_391), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_161, %unsqueeze_393), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_162, %unsqueeze_395), kwargs = {})
#   %convolution_50 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_34, %primals_297, %primals_298, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_405), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_407), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_167, %unsqueeze_409), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_168, %unsqueeze_411), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %add_131), kwargs = {})
#   %add_133 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_132, %relu_27), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_20(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 56)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp5, xmask)
    tl.store(in_out_ptr2 + (x3), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yr/cyrsxhwxkgnk4j7o6sdnxv3lhdk5cojsegk7q242oehiqnpseiv7.py
# Topologically Sorted Source Nodes: [stage3_3_branches_0_1_0, stage3_3_branches_0_1_1, stage3_3_branches_0_1_2, stage3_3_fuse_layers_0_1_1, stage3_3_fuse_layers_0_1_2, add_38, stage3_3_fuse_layers_0_2_1, stage3_3_fuse_layers_0_2_2, add_39], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
# Source node to ATen node mapping:
#   add_38 => add_297
#   add_39 => add_304
#   stage3_3_branches_0_1_0 => convolution_100
#   stage3_3_branches_0_1_1 => add_276, mul_341, mul_342, sub_100
#   stage3_3_branches_0_1_2 => relu_65
#   stage3_3_fuse_layers_0_1_1 => add_292, mul_362, mul_363, sub_107
#   stage3_3_fuse_layers_0_1_2 => _unsafe_index_10
#   stage3_3_fuse_layers_0_2_1 => add_299, mul_369, mul_370, sub_108
#   stage3_3_fuse_layers_0_2_2 => _unsafe_index_11
# Graph fragment:
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_274, %primals_591, %primals_592, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_811), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_813), kwargs = {})
#   %mul_342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_341, %unsqueeze_815), kwargs = {})
#   %add_276 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_342, %unsqueeze_817), kwargs = {})
#   %relu_65 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_276,), kwargs = {})
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_867), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_869), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_362, %unsqueeze_871), kwargs = {})
#   %add_292 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_363, %unsqueeze_873), kwargs = {})
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_292, [None, None, %unsqueeze_184, %convert_element_type_47]), kwargs = {})
#   %add_297 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_65, %_unsafe_index_10), kwargs = {})
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_876), kwargs = {})
#   %mul_369 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_878), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_369, %unsqueeze_880), kwargs = {})
#   %add_299 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_370, %unsqueeze_882), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_299, [None, None, %unsqueeze_322, %convert_element_type_89]), kwargs = {})
#   %add_304 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_297, %_unsafe_index_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_21(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 16)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x3), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
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
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr6 + (tmp28 + 8*tmp24 + 64*x6), None, eviction_policy='evict_last')
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp6
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp9 / tmp34
    tmp36 = tmp35 * tmp11
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp19 + tmp41
    tmp44 = tl.full([XBLOCK], 4, tl.int32)
    tmp45 = tmp43 + tmp44
    tmp46 = tmp43 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp43)
    tmp49 = tmp48 + tmp44
    tmp50 = tmp48 < 0
    tmp51 = tl.where(tmp50, tmp49, tmp48)
    tmp52 = tl.load(in_ptr12 + (tmp51 + 4*tmp47 + 16*x6), None, eviction_policy='evict_last')
    tmp54 = tmp52 - tmp53
    tmp56 = tmp55 + tmp6
    tmp57 = libdevice.sqrt(tmp56)
    tmp58 = tmp9 / tmp57
    tmp59 = tmp58 * tmp11
    tmp60 = tmp54 * tmp59
    tmp62 = tmp60 * tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp42 + tmp64
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(in_out_ptr1 + (x5), tmp65, None)
''', device_str='cuda')


# kernel path: inductor_cache/y6/cy6j6d7tq4w23inihtwfdqgp4vt6oiorrwdvbfqmlzs3ds5ncdlp.py
# Topologically Sorted Source Nodes: [final_layers_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   final_layers_0 => convolution_110
# Graph fragment:
#   %convolution_110 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_70, %primals_649, %primals_650, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_22 = async_compile.triton('triton_poi_fused_convolution_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 47104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 46)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4rukvqd4o5gcuyz4wq4dvtbwztveg2dntu4m7dofwluxofhisun.py
# Topologically Sorted Source Nodes: [deconv_layers_0_0_0], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   deconv_layers_0_0_0 => add_307, add_308, convert_element_type_268, convert_element_type_269, iota_24, mul_378, mul_379
# Graph fragment:
#   %iota_24 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_24, 1), kwargs = {})
#   %add_307 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_378, 0), kwargs = {})
#   %convert_element_type_268 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_307, torch.float32), kwargs = {})
#   %add_308 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_268, 0.0), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_308, 0.5), kwargs = {})
#   %convert_element_type_269 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_379, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_23 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/mt/cmteqd4prramslunlfjeevbi6o5k6jpkijujamjzx4r2u5k7vp6g.py
# Topologically Sorted Source Nodes: [cat_1, deconv_layers_0_0_0], Original ATen: [aten.cat, aten._unsafe_index]
# Source node to ATen node mapping:
#   cat_1 => cat
#   deconv_layers_0_0_0 => _unsafe_index_12
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_70, %convolution_110], 1), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%cat, [None, None, %unsqueeze_892, %convert_element_type_269]), kwargs = {})
triton_poi_fused__unsafe_index_cat_24 = async_compile.triton('triton_poi_fused__unsafe_index_cat_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_cat_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_cat_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 253952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = ((xindex // 1024) % 62)
    x3 = xindex // 63488
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = x2
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 16, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(in_ptr1 + (tmp8 + 16*tmp4 + 256*(x2) + 4096*x3), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp12
    tmp16 = tl.full([1], 62, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*((-16) + x2) + 11776*x3), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp13, tmp14, tmp18)
    tl.store(out_ptr0 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/qq/cqqj2pwg7fttuv6tbdcgipt5cibelpivf2zzgosye2amroi7eerr.py
# Topologically Sorted Source Nodes: [deconv_layers_0_0_2, deconv_layers_0_0_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   deconv_layers_0_0_2 => add_312, mul_383, mul_384, sub_110
#   deconv_layers_0_0_3 => relu_71
# Graph fragment:
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_894), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %unsqueeze_896), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_383, %unsqueeze_898), kwargs = {})
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_384, %unsqueeze_900), kwargs = {})
#   %relu_71 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_312,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 16)
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


# kernel path: inductor_cache/kl/cklxgeoumbl66xcaqux4hdgwc4dafpi4gyq7rkx4be5a5gv4xgrl.py
# Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv1_0, deconv_layers_0_1_0_conv1_1, deconv_layers_0_1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   deconv_layers_0_1_0_conv1_0 => convolution_112
#   deconv_layers_0_1_0_conv1_1 => add_314, mul_386, mul_387, sub_111
#   deconv_layers_0_1_0_conv1_2 => relu_72
# Graph fragment:
#   %convolution_112 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_71, %primals_656, %primals_657, [1, 1], [1, 1], [1, 1], False, [0, 0], 16), kwargs = {})
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_902), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_904), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_386, %unsqueeze_906), kwargs = {})
#   %add_314 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_387, %unsqueeze_908), kwargs = {})
#   %relu_72 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_314,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 16)
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


# kernel path: inductor_cache/eo/ceon2gosxwotsr4ux4qvk4bp5avy2bhle5s6nxotaj2zp3ikpd6f.py
# Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv2_3, deconv_layers_0_1_0_conv2_4, add_40], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_40 => add_321
#   deconv_layers_0_1_0_conv2_3 => convolution_115
#   deconv_layers_0_1_0_conv2_4 => add_320, mul_395, mul_396, sub_114
# Graph fragment:
#   %convolution_115 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_74, %primals_674, %primals_675, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_926), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_928), kwargs = {})
#   %mul_396 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_395, %unsqueeze_930), kwargs = {})
#   %add_320 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_396, %unsqueeze_932), kwargs = {})
#   %add_321 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_320, %relu_71), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), None)
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
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/eu/ceuqz2jc67zymgpdgon6gpxixqctumjnqfxf34hqzibgnz5l6dcj.py
# Topologically Sorted Source Nodes: [final_layers_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   final_layers_1 => convolution_116
# Graph fragment:
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_321, %primals_680, %primals_681, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_28 = async_compile.triton('triton_poi_fused_convolution_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_28(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 23)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_80, (16, ), (1, ))
    assert_size_stride(primals_81, (16, ), (1, ))
    assert_size_stride(primals_82, (16, ), (1, ))
    assert_size_stride(primals_83, (16, ), (1, ))
    assert_size_stride(primals_84, (16, ), (1, ))
    assert_size_stride(primals_85, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (64, ), (1, ))
    assert_size_stride(primals_88, (64, ), (1, ))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_92, (32, ), (1, ))
    assert_size_stride(primals_93, (32, ), (1, ))
    assert_size_stride(primals_94, (32, ), (1, ))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (16, ), (1, ))
    assert_size_stride(primals_99, (16, ), (1, ))
    assert_size_stride(primals_100, (16, ), (1, ))
    assert_size_stride(primals_101, (16, ), (1, ))
    assert_size_stride(primals_102, (16, ), (1, ))
    assert_size_stride(primals_103, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_104, (16, ), (1, ))
    assert_size_stride(primals_105, (16, ), (1, ))
    assert_size_stride(primals_106, (16, ), (1, ))
    assert_size_stride(primals_107, (16, ), (1, ))
    assert_size_stride(primals_108, (16, ), (1, ))
    assert_size_stride(primals_109, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_110, (16, ), (1, ))
    assert_size_stride(primals_111, (16, ), (1, ))
    assert_size_stride(primals_112, (16, ), (1, ))
    assert_size_stride(primals_113, (16, ), (1, ))
    assert_size_stride(primals_114, (16, ), (1, ))
    assert_size_stride(primals_115, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_116, (32, ), (1, ))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, ), (1, ))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, ), (1, ))
    assert_size_stride(primals_121, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_122, (32, ), (1, ))
    assert_size_stride(primals_123, (32, ), (1, ))
    assert_size_stride(primals_124, (32, ), (1, ))
    assert_size_stride(primals_125, (32, ), (1, ))
    assert_size_stride(primals_126, (32, ), (1, ))
    assert_size_stride(primals_127, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_128, (32, ), (1, ))
    assert_size_stride(primals_129, (32, ), (1, ))
    assert_size_stride(primals_130, (32, ), (1, ))
    assert_size_stride(primals_131, (32, ), (1, ))
    assert_size_stride(primals_132, (32, ), (1, ))
    assert_size_stride(primals_133, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_134, (16, ), (1, ))
    assert_size_stride(primals_135, (16, ), (1, ))
    assert_size_stride(primals_136, (16, ), (1, ))
    assert_size_stride(primals_137, (16, ), (1, ))
    assert_size_stride(primals_138, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_139, (16, ), (1, ))
    assert_size_stride(primals_140, (16, ), (1, ))
    assert_size_stride(primals_141, (16, ), (1, ))
    assert_size_stride(primals_142, (16, ), (1, ))
    assert_size_stride(primals_143, (16, ), (1, ))
    assert_size_stride(primals_144, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (16, ), (1, ))
    assert_size_stride(primals_146, (16, ), (1, ))
    assert_size_stride(primals_147, (16, ), (1, ))
    assert_size_stride(primals_148, (16, ), (1, ))
    assert_size_stride(primals_149, (16, ), (1, ))
    assert_size_stride(primals_150, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_151, (32, ), (1, ))
    assert_size_stride(primals_152, (32, ), (1, ))
    assert_size_stride(primals_153, (32, ), (1, ))
    assert_size_stride(primals_154, (32, ), (1, ))
    assert_size_stride(primals_155, (32, ), (1, ))
    assert_size_stride(primals_156, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_157, (32, ), (1, ))
    assert_size_stride(primals_158, (32, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (32, ), (1, ))
    assert_size_stride(primals_162, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_164, (32, ), (1, ))
    assert_size_stride(primals_165, (32, ), (1, ))
    assert_size_stride(primals_166, (32, ), (1, ))
    assert_size_stride(primals_167, (32, ), (1, ))
    assert_size_stride(primals_168, (56, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_169, (56, ), (1, ))
    assert_size_stride(primals_170, (56, ), (1, ))
    assert_size_stride(primals_171, (56, ), (1, ))
    assert_size_stride(primals_172, (56, ), (1, ))
    assert_size_stride(primals_173, (56, ), (1, ))
    assert_size_stride(primals_174, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_175, (16, ), (1, ))
    assert_size_stride(primals_176, (16, ), (1, ))
    assert_size_stride(primals_177, (16, ), (1, ))
    assert_size_stride(primals_178, (16, ), (1, ))
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_180, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_181, (16, ), (1, ))
    assert_size_stride(primals_182, (16, ), (1, ))
    assert_size_stride(primals_183, (16, ), (1, ))
    assert_size_stride(primals_184, (16, ), (1, ))
    assert_size_stride(primals_185, (16, ), (1, ))
    assert_size_stride(primals_186, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_187, (16, ), (1, ))
    assert_size_stride(primals_188, (16, ), (1, ))
    assert_size_stride(primals_189, (16, ), (1, ))
    assert_size_stride(primals_190, (16, ), (1, ))
    assert_size_stride(primals_191, (16, ), (1, ))
    assert_size_stride(primals_192, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (32, ), (1, ))
    assert_size_stride(primals_195, (32, ), (1, ))
    assert_size_stride(primals_196, (32, ), (1, ))
    assert_size_stride(primals_197, (32, ), (1, ))
    assert_size_stride(primals_198, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, ), (1, ))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, ), (1, ))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, ), (1, ))
    assert_size_stride(primals_207, (32, ), (1, ))
    assert_size_stride(primals_208, (32, ), (1, ))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_211, (56, ), (1, ))
    assert_size_stride(primals_212, (56, ), (1, ))
    assert_size_stride(primals_213, (56, ), (1, ))
    assert_size_stride(primals_214, (56, ), (1, ))
    assert_size_stride(primals_215, (56, ), (1, ))
    assert_size_stride(primals_216, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_217, (56, ), (1, ))
    assert_size_stride(primals_218, (56, ), (1, ))
    assert_size_stride(primals_219, (56, ), (1, ))
    assert_size_stride(primals_220, (56, ), (1, ))
    assert_size_stride(primals_221, (56, ), (1, ))
    assert_size_stride(primals_222, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_223, (56, ), (1, ))
    assert_size_stride(primals_224, (56, ), (1, ))
    assert_size_stride(primals_225, (56, ), (1, ))
    assert_size_stride(primals_226, (56, ), (1, ))
    assert_size_stride(primals_227, (56, ), (1, ))
    assert_size_stride(primals_228, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_229, (16, ), (1, ))
    assert_size_stride(primals_230, (16, ), (1, ))
    assert_size_stride(primals_231, (16, ), (1, ))
    assert_size_stride(primals_232, (16, ), (1, ))
    assert_size_stride(primals_233, (16, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_234, (16, ), (1, ))
    assert_size_stride(primals_235, (16, ), (1, ))
    assert_size_stride(primals_236, (16, ), (1, ))
    assert_size_stride(primals_237, (16, ), (1, ))
    assert_size_stride(primals_238, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_239, (16, ), (1, ))
    assert_size_stride(primals_240, (16, ), (1, ))
    assert_size_stride(primals_241, (16, ), (1, ))
    assert_size_stride(primals_242, (16, ), (1, ))
    assert_size_stride(primals_243, (16, ), (1, ))
    assert_size_stride(primals_244, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_245, (16, ), (1, ))
    assert_size_stride(primals_246, (16, ), (1, ))
    assert_size_stride(primals_247, (16, ), (1, ))
    assert_size_stride(primals_248, (16, ), (1, ))
    assert_size_stride(primals_249, (16, ), (1, ))
    assert_size_stride(primals_250, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_251, (32, ), (1, ))
    assert_size_stride(primals_252, (32, ), (1, ))
    assert_size_stride(primals_253, (32, ), (1, ))
    assert_size_stride(primals_254, (32, ), (1, ))
    assert_size_stride(primals_255, (32, ), (1, ))
    assert_size_stride(primals_256, (32, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_257, (32, ), (1, ))
    assert_size_stride(primals_258, (32, ), (1, ))
    assert_size_stride(primals_259, (32, ), (1, ))
    assert_size_stride(primals_260, (32, ), (1, ))
    assert_size_stride(primals_261, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_262, (32, ), (1, ))
    assert_size_stride(primals_263, (32, ), (1, ))
    assert_size_stride(primals_264, (32, ), (1, ))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_268, (16, ), (1, ))
    assert_size_stride(primals_269, (16, ), (1, ))
    assert_size_stride(primals_270, (16, ), (1, ))
    assert_size_stride(primals_271, (16, ), (1, ))
    assert_size_stride(primals_272, (16, ), (1, ))
    assert_size_stride(primals_273, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_274, (16, ), (1, ))
    assert_size_stride(primals_275, (16, ), (1, ))
    assert_size_stride(primals_276, (16, ), (1, ))
    assert_size_stride(primals_277, (16, ), (1, ))
    assert_size_stride(primals_278, (16, ), (1, ))
    assert_size_stride(primals_279, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_280, (16, ), (1, ))
    assert_size_stride(primals_281, (16, ), (1, ))
    assert_size_stride(primals_282, (16, ), (1, ))
    assert_size_stride(primals_283, (16, ), (1, ))
    assert_size_stride(primals_284, (16, ), (1, ))
    assert_size_stride(primals_285, (56, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_286, (56, ), (1, ))
    assert_size_stride(primals_287, (56, ), (1, ))
    assert_size_stride(primals_288, (56, ), (1, ))
    assert_size_stride(primals_289, (56, ), (1, ))
    assert_size_stride(primals_290, (56, ), (1, ))
    assert_size_stride(primals_291, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_292, (32, ), (1, ))
    assert_size_stride(primals_293, (32, ), (1, ))
    assert_size_stride(primals_294, (32, ), (1, ))
    assert_size_stride(primals_295, (32, ), (1, ))
    assert_size_stride(primals_296, (32, ), (1, ))
    assert_size_stride(primals_297, (56, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_298, (56, ), (1, ))
    assert_size_stride(primals_299, (56, ), (1, ))
    assert_size_stride(primals_300, (56, ), (1, ))
    assert_size_stride(primals_301, (56, ), (1, ))
    assert_size_stride(primals_302, (56, ), (1, ))
    assert_size_stride(primals_303, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_304, (56, ), (1, ))
    assert_size_stride(primals_305, (56, ), (1, ))
    assert_size_stride(primals_306, (56, ), (1, ))
    assert_size_stride(primals_307, (56, ), (1, ))
    assert_size_stride(primals_308, (56, ), (1, ))
    assert_size_stride(primals_309, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_310, (16, ), (1, ))
    assert_size_stride(primals_311, (16, ), (1, ))
    assert_size_stride(primals_312, (16, ), (1, ))
    assert_size_stride(primals_313, (16, ), (1, ))
    assert_size_stride(primals_314, (16, ), (1, ))
    assert_size_stride(primals_315, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_316, (16, ), (1, ))
    assert_size_stride(primals_317, (16, ), (1, ))
    assert_size_stride(primals_318, (16, ), (1, ))
    assert_size_stride(primals_319, (16, ), (1, ))
    assert_size_stride(primals_320, (16, ), (1, ))
    assert_size_stride(primals_321, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_322, (16, ), (1, ))
    assert_size_stride(primals_323, (16, ), (1, ))
    assert_size_stride(primals_324, (16, ), (1, ))
    assert_size_stride(primals_325, (16, ), (1, ))
    assert_size_stride(primals_326, (16, ), (1, ))
    assert_size_stride(primals_327, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_328, (32, ), (1, ))
    assert_size_stride(primals_329, (32, ), (1, ))
    assert_size_stride(primals_330, (32, ), (1, ))
    assert_size_stride(primals_331, (32, ), (1, ))
    assert_size_stride(primals_332, (32, ), (1, ))
    assert_size_stride(primals_333, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_334, (32, ), (1, ))
    assert_size_stride(primals_335, (32, ), (1, ))
    assert_size_stride(primals_336, (32, ), (1, ))
    assert_size_stride(primals_337, (32, ), (1, ))
    assert_size_stride(primals_338, (32, ), (1, ))
    assert_size_stride(primals_339, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_340, (32, ), (1, ))
    assert_size_stride(primals_341, (32, ), (1, ))
    assert_size_stride(primals_342, (32, ), (1, ))
    assert_size_stride(primals_343, (32, ), (1, ))
    assert_size_stride(primals_344, (32, ), (1, ))
    assert_size_stride(primals_345, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_346, (56, ), (1, ))
    assert_size_stride(primals_347, (56, ), (1, ))
    assert_size_stride(primals_348, (56, ), (1, ))
    assert_size_stride(primals_349, (56, ), (1, ))
    assert_size_stride(primals_350, (56, ), (1, ))
    assert_size_stride(primals_351, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_352, (56, ), (1, ))
    assert_size_stride(primals_353, (56, ), (1, ))
    assert_size_stride(primals_354, (56, ), (1, ))
    assert_size_stride(primals_355, (56, ), (1, ))
    assert_size_stride(primals_356, (56, ), (1, ))
    assert_size_stride(primals_357, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_358, (56, ), (1, ))
    assert_size_stride(primals_359, (56, ), (1, ))
    assert_size_stride(primals_360, (56, ), (1, ))
    assert_size_stride(primals_361, (56, ), (1, ))
    assert_size_stride(primals_362, (56, ), (1, ))
    assert_size_stride(primals_363, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_364, (16, ), (1, ))
    assert_size_stride(primals_365, (16, ), (1, ))
    assert_size_stride(primals_366, (16, ), (1, ))
    assert_size_stride(primals_367, (16, ), (1, ))
    assert_size_stride(primals_368, (16, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_369, (16, ), (1, ))
    assert_size_stride(primals_370, (16, ), (1, ))
    assert_size_stride(primals_371, (16, ), (1, ))
    assert_size_stride(primals_372, (16, ), (1, ))
    assert_size_stride(primals_373, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_374, (16, ), (1, ))
    assert_size_stride(primals_375, (16, ), (1, ))
    assert_size_stride(primals_376, (16, ), (1, ))
    assert_size_stride(primals_377, (16, ), (1, ))
    assert_size_stride(primals_378, (16, ), (1, ))
    assert_size_stride(primals_379, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_380, (16, ), (1, ))
    assert_size_stride(primals_381, (16, ), (1, ))
    assert_size_stride(primals_382, (16, ), (1, ))
    assert_size_stride(primals_383, (16, ), (1, ))
    assert_size_stride(primals_384, (16, ), (1, ))
    assert_size_stride(primals_385, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_386, (32, ), (1, ))
    assert_size_stride(primals_387, (32, ), (1, ))
    assert_size_stride(primals_388, (32, ), (1, ))
    assert_size_stride(primals_389, (32, ), (1, ))
    assert_size_stride(primals_390, (32, ), (1, ))
    assert_size_stride(primals_391, (32, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_392, (32, ), (1, ))
    assert_size_stride(primals_393, (32, ), (1, ))
    assert_size_stride(primals_394, (32, ), (1, ))
    assert_size_stride(primals_395, (32, ), (1, ))
    assert_size_stride(primals_396, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_397, (32, ), (1, ))
    assert_size_stride(primals_398, (32, ), (1, ))
    assert_size_stride(primals_399, (32, ), (1, ))
    assert_size_stride(primals_400, (32, ), (1, ))
    assert_size_stride(primals_401, (32, ), (1, ))
    assert_size_stride(primals_402, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_403, (16, ), (1, ))
    assert_size_stride(primals_404, (16, ), (1, ))
    assert_size_stride(primals_405, (16, ), (1, ))
    assert_size_stride(primals_406, (16, ), (1, ))
    assert_size_stride(primals_407, (16, ), (1, ))
    assert_size_stride(primals_408, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_409, (16, ), (1, ))
    assert_size_stride(primals_410, (16, ), (1, ))
    assert_size_stride(primals_411, (16, ), (1, ))
    assert_size_stride(primals_412, (16, ), (1, ))
    assert_size_stride(primals_413, (16, ), (1, ))
    assert_size_stride(primals_414, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_415, (16, ), (1, ))
    assert_size_stride(primals_416, (16, ), (1, ))
    assert_size_stride(primals_417, (16, ), (1, ))
    assert_size_stride(primals_418, (16, ), (1, ))
    assert_size_stride(primals_419, (16, ), (1, ))
    assert_size_stride(primals_420, (56, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_421, (56, ), (1, ))
    assert_size_stride(primals_422, (56, ), (1, ))
    assert_size_stride(primals_423, (56, ), (1, ))
    assert_size_stride(primals_424, (56, ), (1, ))
    assert_size_stride(primals_425, (56, ), (1, ))
    assert_size_stride(primals_426, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_427, (32, ), (1, ))
    assert_size_stride(primals_428, (32, ), (1, ))
    assert_size_stride(primals_429, (32, ), (1, ))
    assert_size_stride(primals_430, (32, ), (1, ))
    assert_size_stride(primals_431, (32, ), (1, ))
    assert_size_stride(primals_432, (56, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_433, (56, ), (1, ))
    assert_size_stride(primals_434, (56, ), (1, ))
    assert_size_stride(primals_435, (56, ), (1, ))
    assert_size_stride(primals_436, (56, ), (1, ))
    assert_size_stride(primals_437, (56, ), (1, ))
    assert_size_stride(primals_438, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_439, (56, ), (1, ))
    assert_size_stride(primals_440, (56, ), (1, ))
    assert_size_stride(primals_441, (56, ), (1, ))
    assert_size_stride(primals_442, (56, ), (1, ))
    assert_size_stride(primals_443, (56, ), (1, ))
    assert_size_stride(primals_444, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_445, (16, ), (1, ))
    assert_size_stride(primals_446, (16, ), (1, ))
    assert_size_stride(primals_447, (16, ), (1, ))
    assert_size_stride(primals_448, (16, ), (1, ))
    assert_size_stride(primals_449, (16, ), (1, ))
    assert_size_stride(primals_450, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_451, (16, ), (1, ))
    assert_size_stride(primals_452, (16, ), (1, ))
    assert_size_stride(primals_453, (16, ), (1, ))
    assert_size_stride(primals_454, (16, ), (1, ))
    assert_size_stride(primals_455, (16, ), (1, ))
    assert_size_stride(primals_456, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_457, (16, ), (1, ))
    assert_size_stride(primals_458, (16, ), (1, ))
    assert_size_stride(primals_459, (16, ), (1, ))
    assert_size_stride(primals_460, (16, ), (1, ))
    assert_size_stride(primals_461, (16, ), (1, ))
    assert_size_stride(primals_462, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_463, (32, ), (1, ))
    assert_size_stride(primals_464, (32, ), (1, ))
    assert_size_stride(primals_465, (32, ), (1, ))
    assert_size_stride(primals_466, (32, ), (1, ))
    assert_size_stride(primals_467, (32, ), (1, ))
    assert_size_stride(primals_468, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_469, (32, ), (1, ))
    assert_size_stride(primals_470, (32, ), (1, ))
    assert_size_stride(primals_471, (32, ), (1, ))
    assert_size_stride(primals_472, (32, ), (1, ))
    assert_size_stride(primals_473, (32, ), (1, ))
    assert_size_stride(primals_474, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_475, (32, ), (1, ))
    assert_size_stride(primals_476, (32, ), (1, ))
    assert_size_stride(primals_477, (32, ), (1, ))
    assert_size_stride(primals_478, (32, ), (1, ))
    assert_size_stride(primals_479, (32, ), (1, ))
    assert_size_stride(primals_480, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_481, (56, ), (1, ))
    assert_size_stride(primals_482, (56, ), (1, ))
    assert_size_stride(primals_483, (56, ), (1, ))
    assert_size_stride(primals_484, (56, ), (1, ))
    assert_size_stride(primals_485, (56, ), (1, ))
    assert_size_stride(primals_486, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_487, (56, ), (1, ))
    assert_size_stride(primals_488, (56, ), (1, ))
    assert_size_stride(primals_489, (56, ), (1, ))
    assert_size_stride(primals_490, (56, ), (1, ))
    assert_size_stride(primals_491, (56, ), (1, ))
    assert_size_stride(primals_492, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_493, (56, ), (1, ))
    assert_size_stride(primals_494, (56, ), (1, ))
    assert_size_stride(primals_495, (56, ), (1, ))
    assert_size_stride(primals_496, (56, ), (1, ))
    assert_size_stride(primals_497, (56, ), (1, ))
    assert_size_stride(primals_498, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_499, (16, ), (1, ))
    assert_size_stride(primals_500, (16, ), (1, ))
    assert_size_stride(primals_501, (16, ), (1, ))
    assert_size_stride(primals_502, (16, ), (1, ))
    assert_size_stride(primals_503, (16, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_504, (16, ), (1, ))
    assert_size_stride(primals_505, (16, ), (1, ))
    assert_size_stride(primals_506, (16, ), (1, ))
    assert_size_stride(primals_507, (16, ), (1, ))
    assert_size_stride(primals_508, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_509, (16, ), (1, ))
    assert_size_stride(primals_510, (16, ), (1, ))
    assert_size_stride(primals_511, (16, ), (1, ))
    assert_size_stride(primals_512, (16, ), (1, ))
    assert_size_stride(primals_513, (16, ), (1, ))
    assert_size_stride(primals_514, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_515, (16, ), (1, ))
    assert_size_stride(primals_516, (16, ), (1, ))
    assert_size_stride(primals_517, (16, ), (1, ))
    assert_size_stride(primals_518, (16, ), (1, ))
    assert_size_stride(primals_519, (16, ), (1, ))
    assert_size_stride(primals_520, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_521, (32, ), (1, ))
    assert_size_stride(primals_522, (32, ), (1, ))
    assert_size_stride(primals_523, (32, ), (1, ))
    assert_size_stride(primals_524, (32, ), (1, ))
    assert_size_stride(primals_525, (32, ), (1, ))
    assert_size_stride(primals_526, (32, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_527, (32, ), (1, ))
    assert_size_stride(primals_528, (32, ), (1, ))
    assert_size_stride(primals_529, (32, ), (1, ))
    assert_size_stride(primals_530, (32, ), (1, ))
    assert_size_stride(primals_531, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_532, (32, ), (1, ))
    assert_size_stride(primals_533, (32, ), (1, ))
    assert_size_stride(primals_534, (32, ), (1, ))
    assert_size_stride(primals_535, (32, ), (1, ))
    assert_size_stride(primals_536, (32, ), (1, ))
    assert_size_stride(primals_537, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_538, (16, ), (1, ))
    assert_size_stride(primals_539, (16, ), (1, ))
    assert_size_stride(primals_540, (16, ), (1, ))
    assert_size_stride(primals_541, (16, ), (1, ))
    assert_size_stride(primals_542, (16, ), (1, ))
    assert_size_stride(primals_543, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_544, (16, ), (1, ))
    assert_size_stride(primals_545, (16, ), (1, ))
    assert_size_stride(primals_546, (16, ), (1, ))
    assert_size_stride(primals_547, (16, ), (1, ))
    assert_size_stride(primals_548, (16, ), (1, ))
    assert_size_stride(primals_549, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_550, (16, ), (1, ))
    assert_size_stride(primals_551, (16, ), (1, ))
    assert_size_stride(primals_552, (16, ), (1, ))
    assert_size_stride(primals_553, (16, ), (1, ))
    assert_size_stride(primals_554, (16, ), (1, ))
    assert_size_stride(primals_555, (56, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_556, (56, ), (1, ))
    assert_size_stride(primals_557, (56, ), (1, ))
    assert_size_stride(primals_558, (56, ), (1, ))
    assert_size_stride(primals_559, (56, ), (1, ))
    assert_size_stride(primals_560, (56, ), (1, ))
    assert_size_stride(primals_561, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_562, (32, ), (1, ))
    assert_size_stride(primals_563, (32, ), (1, ))
    assert_size_stride(primals_564, (32, ), (1, ))
    assert_size_stride(primals_565, (32, ), (1, ))
    assert_size_stride(primals_566, (32, ), (1, ))
    assert_size_stride(primals_567, (56, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_568, (56, ), (1, ))
    assert_size_stride(primals_569, (56, ), (1, ))
    assert_size_stride(primals_570, (56, ), (1, ))
    assert_size_stride(primals_571, (56, ), (1, ))
    assert_size_stride(primals_572, (56, ), (1, ))
    assert_size_stride(primals_573, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_574, (56, ), (1, ))
    assert_size_stride(primals_575, (56, ), (1, ))
    assert_size_stride(primals_576, (56, ), (1, ))
    assert_size_stride(primals_577, (56, ), (1, ))
    assert_size_stride(primals_578, (56, ), (1, ))
    assert_size_stride(primals_579, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_580, (16, ), (1, ))
    assert_size_stride(primals_581, (16, ), (1, ))
    assert_size_stride(primals_582, (16, ), (1, ))
    assert_size_stride(primals_583, (16, ), (1, ))
    assert_size_stride(primals_584, (16, ), (1, ))
    assert_size_stride(primals_585, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_586, (16, ), (1, ))
    assert_size_stride(primals_587, (16, ), (1, ))
    assert_size_stride(primals_588, (16, ), (1, ))
    assert_size_stride(primals_589, (16, ), (1, ))
    assert_size_stride(primals_590, (16, ), (1, ))
    assert_size_stride(primals_591, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_592, (16, ), (1, ))
    assert_size_stride(primals_593, (16, ), (1, ))
    assert_size_stride(primals_594, (16, ), (1, ))
    assert_size_stride(primals_595, (16, ), (1, ))
    assert_size_stride(primals_596, (16, ), (1, ))
    assert_size_stride(primals_597, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_598, (32, ), (1, ))
    assert_size_stride(primals_599, (32, ), (1, ))
    assert_size_stride(primals_600, (32, ), (1, ))
    assert_size_stride(primals_601, (32, ), (1, ))
    assert_size_stride(primals_602, (32, ), (1, ))
    assert_size_stride(primals_603, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_604, (32, ), (1, ))
    assert_size_stride(primals_605, (32, ), (1, ))
    assert_size_stride(primals_606, (32, ), (1, ))
    assert_size_stride(primals_607, (32, ), (1, ))
    assert_size_stride(primals_608, (32, ), (1, ))
    assert_size_stride(primals_609, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_610, (32, ), (1, ))
    assert_size_stride(primals_611, (32, ), (1, ))
    assert_size_stride(primals_612, (32, ), (1, ))
    assert_size_stride(primals_613, (32, ), (1, ))
    assert_size_stride(primals_614, (32, ), (1, ))
    assert_size_stride(primals_615, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_616, (56, ), (1, ))
    assert_size_stride(primals_617, (56, ), (1, ))
    assert_size_stride(primals_618, (56, ), (1, ))
    assert_size_stride(primals_619, (56, ), (1, ))
    assert_size_stride(primals_620, (56, ), (1, ))
    assert_size_stride(primals_621, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_622, (56, ), (1, ))
    assert_size_stride(primals_623, (56, ), (1, ))
    assert_size_stride(primals_624, (56, ), (1, ))
    assert_size_stride(primals_625, (56, ), (1, ))
    assert_size_stride(primals_626, (56, ), (1, ))
    assert_size_stride(primals_627, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_628, (56, ), (1, ))
    assert_size_stride(primals_629, (56, ), (1, ))
    assert_size_stride(primals_630, (56, ), (1, ))
    assert_size_stride(primals_631, (56, ), (1, ))
    assert_size_stride(primals_632, (56, ), (1, ))
    assert_size_stride(primals_633, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_634, (16, ), (1, ))
    assert_size_stride(primals_635, (16, ), (1, ))
    assert_size_stride(primals_636, (16, ), (1, ))
    assert_size_stride(primals_637, (16, ), (1, ))
    assert_size_stride(primals_638, (16, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_639, (16, ), (1, ))
    assert_size_stride(primals_640, (16, ), (1, ))
    assert_size_stride(primals_641, (16, ), (1, ))
    assert_size_stride(primals_642, (16, ), (1, ))
    assert_size_stride(primals_643, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_644, (16, ), (1, ))
    assert_size_stride(primals_645, (16, ), (1, ))
    assert_size_stride(primals_646, (16, ), (1, ))
    assert_size_stride(primals_647, (16, ), (1, ))
    assert_size_stride(primals_648, (16, ), (1, ))
    assert_size_stride(primals_649, (46, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_650, (46, ), (1, ))
    assert_size_stride(primals_651, (16, 62, 1, 1), (62, 1, 1, 1))
    assert_size_stride(primals_652, (16, ), (1, ))
    assert_size_stride(primals_653, (16, ), (1, ))
    assert_size_stride(primals_654, (16, ), (1, ))
    assert_size_stride(primals_655, (16, ), (1, ))
    assert_size_stride(primals_656, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_657, (16, ), (1, ))
    assert_size_stride(primals_658, (16, ), (1, ))
    assert_size_stride(primals_659, (16, ), (1, ))
    assert_size_stride(primals_660, (16, ), (1, ))
    assert_size_stride(primals_661, (16, ), (1, ))
    assert_size_stride(primals_662, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_663, (16, ), (1, ))
    assert_size_stride(primals_664, (16, ), (1, ))
    assert_size_stride(primals_665, (16, ), (1, ))
    assert_size_stride(primals_666, (16, ), (1, ))
    assert_size_stride(primals_667, (16, ), (1, ))
    assert_size_stride(primals_668, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_669, (16, ), (1, ))
    assert_size_stride(primals_670, (16, ), (1, ))
    assert_size_stride(primals_671, (16, ), (1, ))
    assert_size_stride(primals_672, (16, ), (1, ))
    assert_size_stride(primals_673, (16, ), (1, ))
    assert_size_stride(primals_674, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_675, (16, ), (1, ))
    assert_size_stride(primals_676, (16, ), (1, ))
    assert_size_stride(primals_677, (16, ), (1, ))
    assert_size_stride(primals_678, (16, ), (1, ))
    assert_size_stride(primals_679, (16, ), (1, ))
    assert_size_stride(primals_680, (23, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_681, (23, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bn1, relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [downsample_conv1_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf2, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [downsample_conv1_0, downsample_conv1_1, downsample_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf3, primals_8, primals_9, primals_10, primals_11, primals_12, buf4, 65536, grid=grid(65536), stream=stream0)
        del primals_12
        del primals_8
        # Topologically Sorted Source Nodes: [downsample_conv1_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [downsample_conv1_3, downsample_conv1_4, downsample_relu1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf6, primals_14, primals_15, primals_16, primals_17, primals_18, buf7, 65536, grid=grid(65536), stream=stream0)
        del primals_14
        del primals_18
        # Topologically Sorted Source Nodes: [downsample_conv2_0], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_19, stride=(1, 1), padding=(3, 3), dilation=(3, 3), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf8, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [downsample_conv2_0, downsample_conv2_1, downsample_conv2_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf9, primals_20, primals_21, primals_22, primals_23, primals_24, buf10, 65536, grid=grid(65536), stream=stream0)
        del primals_20
        del primals_24
        # Topologically Sorted Source Nodes: [downsample_conv2_3], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [downsample_downsample_res_conv_0], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf1, primals_31, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf13, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [downsample_downsample_res_conv_0, downsample_downsample_res_conv_1, downsample_downsample_res_conv_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf14, primals_32, primals_33, primals_34, primals_35, primals_36, buf15, 65536, grid=grid(65536), stream=stream0)
        del primals_32
        del primals_36
        # Topologically Sorted Source Nodes: [downsample_downsample_res_conv_3], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf12 = buf11; del buf11  # reuse
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [downsample_conv2_3, downsample_conv2_4, downsample_downsample_res_conv_3, downsample_downsample_res_conv_4, add_1, downsample_relu2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_2.run(buf12, buf17, buf19, primals_26, primals_38, primals_27, primals_28, primals_29, primals_30, primals_39, primals_40, primals_41, primals_42, 65536, grid=grid(65536), stream=stream0)
        del primals_26
        del primals_30
        del primals_38
        del primals_42
        # Topologically Sorted Source Nodes: [layer1_0_conv1_0], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf20, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf21 = buf20; del buf20  # reuse
        buf22 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_conv1_0, layer1_0_conv1_1, layer1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf21, primals_44, primals_45, primals_46, primals_47, primals_48, buf22, 65536, grid=grid(65536), stream=stream0)
        del primals_44
        del primals_48
        # Topologically Sorted Source Nodes: [layer1_0_conv1_3], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_conv1_3, layer1_0_conv1_4, add_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3.run(buf24, primals_50, primals_51, primals_52, primals_53, primals_54, buf19, buf25, 65536, grid=grid(65536), stream=stream0)
        del primals_50
        del primals_54
        # Topologically Sorted Source Nodes: [layer1_1_conv1_0], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf26, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_1_conv1_0, layer1_1_conv1_1, layer1_1_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf27, primals_56, primals_57, primals_58, primals_59, primals_60, buf28, 65536, grid=grid(65536), stream=stream0)
        del primals_56
        del primals_60
        # Topologically Sorted Source Nodes: [layer1_1_conv1_3], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf30 = buf29; del buf29  # reuse
        buf31 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_1_conv1_3, layer1_1_conv1_4, add_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3.run(buf30, primals_62, primals_63, primals_64, primals_65, primals_66, buf25, buf31, 65536, grid=grid(65536), stream=stream0)
        del primals_62
        del primals_66
        # Topologically Sorted Source Nodes: [cut1_0], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cut1_0, cut1_1, cut1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf33, primals_68, primals_69, primals_70, primals_71, primals_72, buf34, 65536, grid=grid(65536), stream=stream0)
        del primals_68
        del primals_72
        # Topologically Sorted Source Nodes: [transition1_0_0_0], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf35, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [transition1_0_0_0, transition1_0_0_1, transition1_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf36, primals_74, primals_75, primals_76, primals_77, primals_78, buf37, 65536, grid=grid(65536), stream=stream0)
        del primals_74
        del primals_78
        # Topologically Sorted Source Nodes: [transition1_0_0_3], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf39 = buf38; del buf38  # reuse
        buf40 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [transition1_0_0_3, transition1_0_1, transition1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf39, primals_80, primals_81, primals_82, primals_83, primals_84, buf40, 16384, grid=grid(16384), stream=stream0)
        del primals_80
        del primals_84
        # Topologically Sorted Source Nodes: [transition1_1_0_0_0], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf34, primals_85, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf41, (4, 64, 8, 8), (4096, 64, 8, 1))
        buf42 = buf41; del buf41  # reuse
        buf43 = empty_strided_cuda((4, 64, 8, 8), (4096, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [transition1_1_0_0_0, transition1_1_0_0_1, transition1_1_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf42, primals_86, primals_87, primals_88, primals_89, primals_90, buf43, 16384, grid=grid(16384), stream=stream0)
        del primals_86
        del primals_90
        # Topologically Sorted Source Nodes: [transition1_1_0_0_3], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf45 = buf44; del buf44  # reuse
        buf46 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [transition1_1_0_0_3, transition1_1_0_1, transition1_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf45, primals_92, primals_93, primals_94, primals_95, primals_96, buf46, 8192, grid=grid(8192), stream=stream0)
        del primals_92
        del primals_96
        # Topologically Sorted Source Nodes: [stage2_0_branches_0_0_conv1_0], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf40, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf47, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf48 = buf47; del buf47  # reuse
        buf49 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_branches_0_0_conv1_0, stage2_0_branches_0_0_conv1_1, stage2_0_branches_0_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf48, primals_98, primals_99, primals_100, primals_101, primals_102, buf49, 16384, grid=grid(16384), stream=stream0)
        del primals_102
        del primals_98
        # Topologically Sorted Source Nodes: [stage2_0_branches_0_0_conv1_3], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf51 = buf50; del buf50  # reuse
        buf52 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_branches_0_0_conv1_3, stage2_0_branches_0_0_conv1_4, add_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7.run(buf51, primals_104, primals_105, primals_106, primals_107, primals_108, buf40, buf52, 16384, grid=grid(16384), stream=stream0)
        del primals_104
        del primals_108
        # Topologically Sorted Source Nodes: [stage2_0_branches_0_1_0], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [stage2_0_branches_1_0_conv1_0], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf46, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf56, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf57 = buf56; del buf56  # reuse
        buf58 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_branches_1_0_conv1_0, stage2_0_branches_1_0_conv1_1, stage2_0_branches_1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf57, primals_116, primals_117, primals_118, primals_119, primals_120, buf58, 8192, grid=grid(8192), stream=stream0)
        del primals_116
        del primals_120
        # Topologically Sorted Source Nodes: [stage2_0_branches_1_0_conv1_3], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf60 = buf59; del buf59  # reuse
        buf61 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_branches_1_0_conv1_3, stage2_0_branches_1_0_conv1_4, add_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf60, primals_122, primals_123, primals_124, primals_125, primals_126, buf46, buf61, 8192, grid=grid(8192), stream=stream0)
        del primals_122
        del primals_126
        # Topologically Sorted Source Nodes: [stage2_0_branches_1_1_0], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf63 = buf62; del buf62  # reuse
        buf64 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_branches_1_1_0, stage2_0_branches_1_1_1, stage2_0_branches_1_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf63, primals_128, primals_129, primals_130, primals_131, primals_132, buf64, 8192, grid=grid(8192), stream=stream0)
        del primals_128
        del primals_132
        # Topologically Sorted Source Nodes: [stage2_0_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf66 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [stage2_0_fuse_layers_0_1_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_9.run(buf66, 16, grid=grid(16), stream=stream0)
        buf54 = buf53; del buf53  # reuse
        buf55 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf67 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_branches_0_1_0, stage2_0_branches_0_1_1, stage2_0_branches_0_1_2, stage2_0_fuse_layers_0_1_1, stage2_0_fuse_layers_0_1_2, add_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_10.run(buf54, primals_110, primals_111, primals_112, primals_113, primals_114, buf66, buf65, primals_134, primals_135, primals_136, primals_137, buf55, buf67, 16384, grid=grid(16384), stream=stream0)
        del primals_110
        del primals_114
        del primals_137
        # Topologically Sorted Source Nodes: [stage2_0_relu_cbrs_0_0_0], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf69 = buf68; del buf68  # reuse
        buf70 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_relu_cbrs_0_0_0, stage2_0_relu_cbrs_0_0_1, stage2_0_relu_cbrs_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf69, primals_139, primals_140, primals_141, primals_142, primals_143, buf70, 16384, grid=grid(16384), stream=stream0)
        del primals_139
        del primals_143
        # Topologically Sorted Source Nodes: [stage2_0_fuse_layers_1_0_0_0_0], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf55, primals_144, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf71, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf72 = buf71; del buf71  # reuse
        buf73 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_fuse_layers_1_0_0_0_0, stage2_0_fuse_layers_1_0_0_0_1, stage2_0_fuse_layers_1_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf72, primals_145, primals_146, primals_147, primals_148, primals_149, buf73, 4096, grid=grid(4096), stream=stream0)
        del primals_145
        del primals_149
        # Topologically Sorted Source Nodes: [stage2_0_fuse_layers_1_0_0_0_3], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf75 = buf74; del buf74  # reuse
        buf76 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_fuse_layers_1_0_0_0_3, stage2_0_fuse_layers_1_0_0_1, add_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf75, primals_151, primals_152, primals_153, primals_154, primals_155, buf64, buf76, 8192, grid=grid(8192), stream=stream0)
        del primals_151
        del primals_155
        # Topologically Sorted Source Nodes: [stage2_0_relu_cbrs_1_0_0], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf78 = buf77; del buf77  # reuse
        buf79 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage2_0_relu_cbrs_1_0_0, stage2_0_relu_cbrs_1_0_1, stage2_0_relu_cbrs_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf78, primals_157, primals_158, primals_159, primals_160, primals_161, buf79, 8192, grid=grid(8192), stream=stream0)
        del primals_157
        del primals_161
        # Topologically Sorted Source Nodes: [transition2_2_0_0_0], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_162, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf80, (4, 32, 4, 4), (512, 16, 4, 1))
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [transition2_2_0_0_0, transition2_2_0_0_1, transition2_2_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(buf81, primals_163, primals_164, primals_165, primals_166, primals_167, buf82, 2048, grid=grid(2048), stream=stream0)
        del primals_163
        del primals_167
        # Topologically Sorted Source Nodes: [transition2_2_0_0_3], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 56, 4, 4), (896, 16, 4, 1))
        buf84 = buf83; del buf83  # reuse
        buf85 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [transition2_2_0_0_3, transition2_2_0_1, transition2_2_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf84, primals_169, primals_170, primals_171, primals_172, primals_173, buf85, 3584, grid=grid(3584), stream=stream0)
        del primals_169
        del primals_173
        # Topologically Sorted Source Nodes: [stage3_0_branches_0_0_conv1_0], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf70, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf86, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf87 = buf86; del buf86  # reuse
        buf88 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_0_0_conv1_0, stage3_0_branches_0_0_conv1_1, stage3_0_branches_0_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf87, primals_175, primals_176, primals_177, primals_178, primals_179, buf88, 16384, grid=grid(16384), stream=stream0)
        del primals_175
        del primals_179
        # Topologically Sorted Source Nodes: [stage3_0_branches_0_0_conv1_3], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf90 = buf89; del buf89  # reuse
        buf91 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_0_0_conv1_3, stage3_0_branches_0_0_conv1_4, add_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7.run(buf90, primals_181, primals_182, primals_183, primals_184, primals_185, buf70, buf91, 16384, grid=grid(16384), stream=stream0)
        del primals_181
        del primals_185
        # Topologically Sorted Source Nodes: [stage3_0_branches_0_1_0], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [stage3_0_branches_1_0_conv1_0], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf79, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf95, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_1_0_conv1_0, stage3_0_branches_1_0_conv1_1, stage3_0_branches_1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf96, primals_193, primals_194, primals_195, primals_196, primals_197, buf97, 8192, grid=grid(8192), stream=stream0)
        del primals_193
        del primals_197
        # Topologically Sorted Source Nodes: [stage3_0_branches_1_0_conv1_3], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf99 = buf98; del buf98  # reuse
        buf100 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_1_0_conv1_3, stage3_0_branches_1_0_conv1_4, add_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf99, primals_199, primals_200, primals_201, primals_202, primals_203, buf79, buf100, 8192, grid=grid(8192), stream=stream0)
        del primals_199
        del primals_203
        # Topologically Sorted Source Nodes: [stage3_0_branches_1_1_0], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf102 = buf101; del buf101  # reuse
        buf103 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_1_1_0, stage3_0_branches_1_1_1, stage3_0_branches_1_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf102, primals_205, primals_206, primals_207, primals_208, primals_209, buf103, 8192, grid=grid(8192), stream=stream0)
        del primals_205
        del primals_209
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf103, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 16, 8, 8), (1024, 64, 8, 1))
        # Topologically Sorted Source Nodes: [stage3_0_branches_2_0_conv1_0], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf85, primals_210, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf104, (4, 56, 4, 4), (896, 16, 4, 1))
        buf105 = buf104; del buf104  # reuse
        buf106 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_2_0_conv1_0, stage3_0_branches_2_0_conv1_1, stage3_0_branches_2_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf105, primals_211, primals_212, primals_213, primals_214, primals_215, buf106, 3584, grid=grid(3584), stream=stream0)
        del primals_211
        del primals_215
        # Topologically Sorted Source Nodes: [stage3_0_branches_2_0_conv1_3], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 56, 4, 4), (896, 16, 4, 1))
        buf108 = buf107; del buf107  # reuse
        buf109 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_2_0_conv1_3, stage3_0_branches_2_0_conv1_4, add_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14.run(buf108, primals_217, primals_218, primals_219, primals_220, primals_221, buf85, buf109, 3584, grid=grid(3584), stream=stream0)
        del primals_217
        del primals_221
        # Topologically Sorted Source Nodes: [stage3_0_branches_2_1_0], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 56, 4, 4), (896, 16, 4, 1))
        buf111 = buf110; del buf110  # reuse
        buf112 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_2_1_0, stage3_0_branches_2_1_1, stage3_0_branches_2_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf111, primals_223, primals_224, primals_225, primals_226, primals_227, buf112, 3584, grid=grid(3584), stream=stream0)
        del primals_223
        del primals_227
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf112, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 16, 4, 4), (256, 16, 4, 1))
        buf115 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [stage2_0_fuse_layers_0_1_2, stage3_0_fuse_layers_0_2_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_15.run(buf115, 16, grid=grid(16), stream=stream0)
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf116 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_branches_0_1_0, stage3_0_branches_0_1_1, stage3_0_branches_0_1_2, stage3_0_fuse_layers_0_1_1, stage3_0_fuse_layers_0_1_2, add_11, stage3_0_fuse_layers_0_2_1, stage3_0_fuse_layers_0_2_2, add_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_16.run(buf93, primals_187, primals_188, primals_189, primals_190, primals_191, buf66, buf113, primals_229, primals_230, primals_231, primals_232, buf115, buf114, primals_234, primals_235, primals_236, primals_237, buf94, buf116, 16384, grid=grid(16384), stream=stream0)
        del primals_187
        del primals_191
        del primals_232
        del primals_237
        # Topologically Sorted Source Nodes: [stage3_0_relu_cbrs_0_0_0], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf118 = buf117; del buf117  # reuse
        buf119 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_relu_cbrs_0_0_0, stage3_0_relu_cbrs_0_0_1, stage3_0_relu_cbrs_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf118, primals_239, primals_240, primals_241, primals_242, primals_243, buf119, 16384, grid=grid(16384), stream=stream0)
        del primals_239
        del primals_243
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_0_0_0_0], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf94, primals_244, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf120, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf121 = buf120; del buf120  # reuse
        buf122 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_0_0_0_0, stage3_0_fuse_layers_1_0_0_0_1, stage3_0_fuse_layers_1_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf121, primals_245, primals_246, primals_247, primals_248, primals_249, buf122, 4096, grid=grid(4096), stream=stream0)
        del primals_245
        del primals_249
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_0_0_0_3], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf112, primals_256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 32, 4, 4), (512, 16, 4, 1))
        buf126 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_2_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_17.run(buf126, 8, grid=grid(8), stream=stream0)
        buf124 = buf123; del buf123  # reuse
        buf127 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_1_0_0_0_3, stage3_0_fuse_layers_1_0_0_1, add_13, stage3_0_fuse_layers_1_2_1, stage3_0_fuse_layers_1_2_2, add_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_18.run(buf124, primals_251, primals_252, primals_253, primals_254, primals_255, buf103, buf126, buf125, primals_257, primals_258, primals_259, primals_260, buf127, 8192, grid=grid(8192), stream=stream0)
        del primals_251
        del primals_255
        del primals_260
        # Topologically Sorted Source Nodes: [stage3_0_relu_cbrs_1_0_0], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf129 = buf128; del buf128  # reuse
        buf130 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_relu_cbrs_1_0_0, stage3_0_relu_cbrs_1_0_1, stage3_0_relu_cbrs_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf129, primals_262, primals_263, primals_264, primals_265, primals_266, buf130, 8192, grid=grid(8192), stream=stream0)
        del primals_262
        del primals_266
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_0_0_0], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf94, primals_267, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf131, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf132 = buf131; del buf131  # reuse
        buf133 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_0_0_0, stage3_0_fuse_layers_2_0_0_0_1, stage3_0_fuse_layers_2_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf132, primals_268, primals_269, primals_270, primals_271, primals_272, buf133, 4096, grid=grid(4096), stream=stream0)
        del primals_268
        del primals_272
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_0_0_3], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf135 = buf134; del buf134  # reuse
        buf136 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_0_0_3, stage3_0_fuse_layers_2_0_0_1, stage3_0_fuse_layers_2_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf135, primals_274, primals_275, primals_276, primals_277, primals_278, buf136, 4096, grid=grid(4096), stream=stream0)
        del primals_274
        del primals_278
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_1_0_0], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_279, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf137, (4, 16, 4, 4), (256, 16, 4, 1))
        buf138 = buf137; del buf137  # reuse
        buf139 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_1_0_0, stage3_0_fuse_layers_2_0_1_0_1, stage3_0_fuse_layers_2_0_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf138, primals_280, primals_281, primals_282, primals_283, primals_284, buf139, 1024, grid=grid(1024), stream=stream0)
        del primals_280
        del primals_284
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_1_0_3], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 56, 4, 4), (896, 16, 4, 1))
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_1_0_0_0], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf103, primals_291, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf142, (4, 32, 4, 4), (512, 16, 4, 1))
        buf143 = buf142; del buf142  # reuse
        buf144 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_1_0_0_0, stage3_0_fuse_layers_2_1_0_0_1, stage3_0_fuse_layers_2_1_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(buf143, primals_292, primals_293, primals_294, primals_295, primals_296, buf144, 2048, grid=grid(2048), stream=stream0)
        del primals_292
        del primals_296
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_1_0_0_3], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 56, 4, 4), (896, 16, 4, 1))
        buf141 = buf140; del buf140  # reuse
        buf146 = buf145; del buf145  # reuse
        buf147 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [stage3_0_fuse_layers_2_0_1_0_3, stage3_0_fuse_layers_2_0_1_1, stage3_0_fuse_layers_2_1_0_0_3, stage3_0_fuse_layers_2_1_0_1, add_15, add_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_20.run(buf141, buf146, buf148, primals_286, primals_298, primals_287, primals_288, primals_289, primals_290, primals_299, primals_300, primals_301, primals_302, buf112, 3584, grid=grid(3584), stream=stream0)
        del primals_286
        del primals_290
        del primals_298
        del primals_302
        # Topologically Sorted Source Nodes: [stage3_0_relu_cbrs_2_0_0], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 56, 4, 4), (896, 16, 4, 1))
        buf150 = buf149; del buf149  # reuse
        buf151 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_0_relu_cbrs_2_0_0, stage3_0_relu_cbrs_2_0_1, stage3_0_relu_cbrs_2_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf150, primals_304, primals_305, primals_306, primals_307, primals_308, buf151, 3584, grid=grid(3584), stream=stream0)
        del primals_304
        del primals_308
        # Topologically Sorted Source Nodes: [stage3_1_branches_0_0_conv1_0], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf119, primals_309, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf152, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf153 = buf152; del buf152  # reuse
        buf154 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_0_0_conv1_0, stage3_1_branches_0_0_conv1_1, stage3_1_branches_0_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf153, primals_310, primals_311, primals_312, primals_313, primals_314, buf154, 16384, grid=grid(16384), stream=stream0)
        del primals_310
        del primals_314
        # Topologically Sorted Source Nodes: [stage3_1_branches_0_0_conv1_3], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf156 = buf155; del buf155  # reuse
        buf157 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_0_0_conv1_3, stage3_1_branches_0_0_conv1_4, add_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7.run(buf156, primals_316, primals_317, primals_318, primals_319, primals_320, buf119, buf157, 16384, grid=grid(16384), stream=stream0)
        del primals_316
        del primals_320
        # Topologically Sorted Source Nodes: [stage3_1_branches_0_1_0], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [stage3_1_branches_1_0_conv1_0], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf130, primals_327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf161, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf162 = buf161; del buf161  # reuse
        buf163 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_1_0_conv1_0, stage3_1_branches_1_0_conv1_1, stage3_1_branches_1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf162, primals_328, primals_329, primals_330, primals_331, primals_332, buf163, 8192, grid=grid(8192), stream=stream0)
        del primals_328
        del primals_332
        # Topologically Sorted Source Nodes: [stage3_1_branches_1_0_conv1_3], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_333, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf165 = buf164; del buf164  # reuse
        buf166 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_1_0_conv1_3, stage3_1_branches_1_0_conv1_4, add_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf165, primals_334, primals_335, primals_336, primals_337, primals_338, buf130, buf166, 8192, grid=grid(8192), stream=stream0)
        del primals_334
        del primals_338
        # Topologically Sorted Source Nodes: [stage3_1_branches_1_1_0], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_339, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf168 = buf167; del buf167  # reuse
        buf169 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_1_1_0, stage3_1_branches_1_1_1, stage3_1_branches_1_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf168, primals_340, primals_341, primals_342, primals_343, primals_344, buf169, 8192, grid=grid(8192), stream=stream0)
        del primals_340
        del primals_344
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf169, primals_363, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 16, 8, 8), (1024, 64, 8, 1))
        # Topologically Sorted Source Nodes: [stage3_1_branches_2_0_conv1_0], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf151, primals_345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf170, (4, 56, 4, 4), (896, 16, 4, 1))
        buf171 = buf170; del buf170  # reuse
        buf172 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_2_0_conv1_0, stage3_1_branches_2_0_conv1_1, stage3_1_branches_2_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf171, primals_346, primals_347, primals_348, primals_349, primals_350, buf172, 3584, grid=grid(3584), stream=stream0)
        del primals_346
        del primals_350
        # Topologically Sorted Source Nodes: [stage3_1_branches_2_0_conv1_3], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_351, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 56, 4, 4), (896, 16, 4, 1))
        buf174 = buf173; del buf173  # reuse
        buf175 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_2_0_conv1_3, stage3_1_branches_2_0_conv1_4, add_19], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14.run(buf174, primals_352, primals_353, primals_354, primals_355, primals_356, buf151, buf175, 3584, grid=grid(3584), stream=stream0)
        del primals_352
        del primals_356
        # Topologically Sorted Source Nodes: [stage3_1_branches_2_1_0], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 56, 4, 4), (896, 16, 4, 1))
        buf177 = buf176; del buf176  # reuse
        buf178 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_2_1_0, stage3_1_branches_2_1_1, stage3_1_branches_2_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf177, primals_358, primals_359, primals_360, primals_361, primals_362, buf178, 3584, grid=grid(3584), stream=stream0)
        del primals_358
        del primals_362
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf178, primals_368, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 16, 4, 4), (256, 16, 4, 1))
        buf159 = buf158; del buf158  # reuse
        buf160 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf181 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_branches_0_1_0, stage3_1_branches_0_1_1, stage3_1_branches_0_1_2, stage3_1_fuse_layers_0_1_1, stage3_1_fuse_layers_0_1_2, add_20, stage3_1_fuse_layers_0_2_1, stage3_1_fuse_layers_0_2_2, add_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_16.run(buf159, primals_322, primals_323, primals_324, primals_325, primals_326, buf66, buf179, primals_364, primals_365, primals_366, primals_367, buf115, buf180, primals_369, primals_370, primals_371, primals_372, buf160, buf181, 16384, grid=grid(16384), stream=stream0)
        del primals_322
        del primals_326
        del primals_367
        del primals_372
        # Topologically Sorted Source Nodes: [stage3_1_relu_cbrs_0_0_0], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_373, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf183 = buf182; del buf182  # reuse
        buf184 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_relu_cbrs_0_0_0, stage3_1_relu_cbrs_0_0_1, stage3_1_relu_cbrs_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf183, primals_374, primals_375, primals_376, primals_377, primals_378, buf184, 16384, grid=grid(16384), stream=stream0)
        del primals_374
        del primals_378
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_1_0_0_0_0], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf160, primals_379, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf185, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf186 = buf185; del buf185  # reuse
        buf187 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_1_0_0_0_0, stage3_1_fuse_layers_1_0_0_0_1, stage3_1_fuse_layers_1_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf186, primals_380, primals_381, primals_382, primals_383, primals_384, buf187, 4096, grid=grid(4096), stream=stream0)
        del primals_380
        del primals_384
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_1_0_0_0_3], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_385, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf178, primals_391, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 32, 4, 4), (512, 16, 4, 1))
        buf189 = buf188; del buf188  # reuse
        buf191 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_1_0_0_0_3, stage3_1_fuse_layers_1_0_0_1, add_22, stage3_1_fuse_layers_1_2_1, stage3_1_fuse_layers_1_2_2, add_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_18.run(buf189, primals_386, primals_387, primals_388, primals_389, primals_390, buf169, buf126, buf190, primals_392, primals_393, primals_394, primals_395, buf191, 8192, grid=grid(8192), stream=stream0)
        del primals_386
        del primals_390
        del primals_395
        # Topologically Sorted Source Nodes: [stage3_1_relu_cbrs_1_0_0], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_396, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf193 = buf192; del buf192  # reuse
        buf194 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_relu_cbrs_1_0_0, stage3_1_relu_cbrs_1_0_1, stage3_1_relu_cbrs_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf193, primals_397, primals_398, primals_399, primals_400, primals_401, buf194, 8192, grid=grid(8192), stream=stream0)
        del primals_397
        del primals_401
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_0_0_0], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf160, primals_402, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf195, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf196 = buf195; del buf195  # reuse
        buf197 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_0_0_0, stage3_1_fuse_layers_2_0_0_0_1, stage3_1_fuse_layers_2_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf196, primals_403, primals_404, primals_405, primals_406, primals_407, buf197, 4096, grid=grid(4096), stream=stream0)
        del primals_403
        del primals_407
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_0_0_3], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_408, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf199 = buf198; del buf198  # reuse
        buf200 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_0_0_3, stage3_1_fuse_layers_2_0_0_1, stage3_1_fuse_layers_2_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf199, primals_409, primals_410, primals_411, primals_412, primals_413, buf200, 4096, grid=grid(4096), stream=stream0)
        del primals_409
        del primals_413
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_1_0_0], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_414, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf201, (4, 16, 4, 4), (256, 16, 4, 1))
        buf202 = buf201; del buf201  # reuse
        buf203 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_1_0_0, stage3_1_fuse_layers_2_0_1_0_1, stage3_1_fuse_layers_2_0_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf202, primals_415, primals_416, primals_417, primals_418, primals_419, buf203, 1024, grid=grid(1024), stream=stream0)
        del primals_415
        del primals_419
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_1_0_3], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_420, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 56, 4, 4), (896, 16, 4, 1))
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_1_0_0_0], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf169, primals_426, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf206, (4, 32, 4, 4), (512, 16, 4, 1))
        buf207 = buf206; del buf206  # reuse
        buf208 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_1_0_0_0, stage3_1_fuse_layers_2_1_0_0_1, stage3_1_fuse_layers_2_1_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(buf207, primals_427, primals_428, primals_429, primals_430, primals_431, buf208, 2048, grid=grid(2048), stream=stream0)
        del primals_427
        del primals_431
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_1_0_0_3], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 56, 4, 4), (896, 16, 4, 1))
        buf205 = buf204; del buf204  # reuse
        buf210 = buf209; del buf209  # reuse
        buf211 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [stage3_1_fuse_layers_2_0_1_0_3, stage3_1_fuse_layers_2_0_1_1, stage3_1_fuse_layers_2_1_0_0_3, stage3_1_fuse_layers_2_1_0_1, add_24, add_25], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_20.run(buf205, buf210, buf212, primals_421, primals_433, primals_422, primals_423, primals_424, primals_425, primals_434, primals_435, primals_436, primals_437, buf178, 3584, grid=grid(3584), stream=stream0)
        del primals_421
        del primals_425
        del primals_433
        del primals_437
        # Topologically Sorted Source Nodes: [stage3_1_relu_cbrs_2_0_0], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_438, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 56, 4, 4), (896, 16, 4, 1))
        buf214 = buf213; del buf213  # reuse
        buf215 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_1_relu_cbrs_2_0_0, stage3_1_relu_cbrs_2_0_1, stage3_1_relu_cbrs_2_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf214, primals_439, primals_440, primals_441, primals_442, primals_443, buf215, 3584, grid=grid(3584), stream=stream0)
        del primals_439
        del primals_443
        # Topologically Sorted Source Nodes: [stage3_2_branches_0_0_conv1_0], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf184, primals_444, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf216, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf217 = buf216; del buf216  # reuse
        buf218 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_0_0_conv1_0, stage3_2_branches_0_0_conv1_1, stage3_2_branches_0_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf217, primals_445, primals_446, primals_447, primals_448, primals_449, buf218, 16384, grid=grid(16384), stream=stream0)
        del primals_445
        del primals_449
        # Topologically Sorted Source Nodes: [stage3_2_branches_0_0_conv1_3], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_450, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf220 = buf219; del buf219  # reuse
        buf221 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_0_0_conv1_3, stage3_2_branches_0_0_conv1_4, add_26], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7.run(buf220, primals_451, primals_452, primals_453, primals_454, primals_455, buf184, buf221, 16384, grid=grid(16384), stream=stream0)
        del primals_451
        del primals_455
        # Topologically Sorted Source Nodes: [stage3_2_branches_0_1_0], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_456, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [stage3_2_branches_1_0_conv1_0], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf194, primals_462, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf225, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf226 = buf225; del buf225  # reuse
        buf227 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_1_0_conv1_0, stage3_2_branches_1_0_conv1_1, stage3_2_branches_1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf226, primals_463, primals_464, primals_465, primals_466, primals_467, buf227, 8192, grid=grid(8192), stream=stream0)
        del primals_463
        del primals_467
        # Topologically Sorted Source Nodes: [stage3_2_branches_1_0_conv1_3], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf229 = buf228; del buf228  # reuse
        buf230 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_1_0_conv1_3, stage3_2_branches_1_0_conv1_4, add_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf229, primals_469, primals_470, primals_471, primals_472, primals_473, buf194, buf230, 8192, grid=grid(8192), stream=stream0)
        del primals_469
        del primals_473
        # Topologically Sorted Source Nodes: [stage3_2_branches_1_1_0], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_474, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf232 = buf231; del buf231  # reuse
        buf233 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_1_1_0, stage3_2_branches_1_1_1, stage3_2_branches_1_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf232, primals_475, primals_476, primals_477, primals_478, primals_479, buf233, 8192, grid=grid(8192), stream=stream0)
        del primals_475
        del primals_479
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf233, primals_498, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 16, 8, 8), (1024, 64, 8, 1))
        # Topologically Sorted Source Nodes: [stage3_2_branches_2_0_conv1_0], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf215, primals_480, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf234, (4, 56, 4, 4), (896, 16, 4, 1))
        buf235 = buf234; del buf234  # reuse
        buf236 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_2_0_conv1_0, stage3_2_branches_2_0_conv1_1, stage3_2_branches_2_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf235, primals_481, primals_482, primals_483, primals_484, primals_485, buf236, 3584, grid=grid(3584), stream=stream0)
        del primals_481
        del primals_485
        # Topologically Sorted Source Nodes: [stage3_2_branches_2_0_conv1_3], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_486, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 56, 4, 4), (896, 16, 4, 1))
        buf238 = buf237; del buf237  # reuse
        buf239 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_2_0_conv1_3, stage3_2_branches_2_0_conv1_4, add_28], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14.run(buf238, primals_487, primals_488, primals_489, primals_490, primals_491, buf215, buf239, 3584, grid=grid(3584), stream=stream0)
        del primals_487
        del primals_491
        # Topologically Sorted Source Nodes: [stage3_2_branches_2_1_0], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 56, 4, 4), (896, 16, 4, 1))
        buf241 = buf240; del buf240  # reuse
        buf242 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_2_1_0, stage3_2_branches_2_1_1, stage3_2_branches_2_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf241, primals_493, primals_494, primals_495, primals_496, primals_497, buf242, 3584, grid=grid(3584), stream=stream0)
        del primals_493
        del primals_497
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf242, primals_503, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 16, 4, 4), (256, 16, 4, 1))
        buf223 = buf222; del buf222  # reuse
        buf224 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf245 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_branches_0_1_0, stage3_2_branches_0_1_1, stage3_2_branches_0_1_2, stage3_2_fuse_layers_0_1_1, stage3_2_fuse_layers_0_1_2, add_29, stage3_2_fuse_layers_0_2_1, stage3_2_fuse_layers_0_2_2, add_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_16.run(buf223, primals_457, primals_458, primals_459, primals_460, primals_461, buf66, buf243, primals_499, primals_500, primals_501, primals_502, buf115, buf244, primals_504, primals_505, primals_506, primals_507, buf224, buf245, 16384, grid=grid(16384), stream=stream0)
        del primals_457
        del primals_461
        del primals_502
        del primals_507
        # Topologically Sorted Source Nodes: [stage3_2_relu_cbrs_0_0_0], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_508, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf247 = buf246; del buf246  # reuse
        buf248 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_relu_cbrs_0_0_0, stage3_2_relu_cbrs_0_0_1, stage3_2_relu_cbrs_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf247, primals_509, primals_510, primals_511, primals_512, primals_513, buf248, 16384, grid=grid(16384), stream=stream0)
        del primals_509
        del primals_513
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_1_0_0_0_0], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf224, primals_514, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf249, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf250 = buf249; del buf249  # reuse
        buf251 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_1_0_0_0_0, stage3_2_fuse_layers_1_0_0_0_1, stage3_2_fuse_layers_1_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf250, primals_515, primals_516, primals_517, primals_518, primals_519, buf251, 4096, grid=grid(4096), stream=stream0)
        del primals_515
        del primals_519
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_1_0_0_0_3], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_520, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf242, primals_526, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 32, 4, 4), (512, 16, 4, 1))
        buf253 = buf252; del buf252  # reuse
        buf255 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_1_0_0_0_3, stage3_2_fuse_layers_1_0_0_1, add_31, stage3_2_fuse_layers_1_2_1, stage3_2_fuse_layers_1_2_2, add_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_18.run(buf253, primals_521, primals_522, primals_523, primals_524, primals_525, buf233, buf126, buf254, primals_527, primals_528, primals_529, primals_530, buf255, 8192, grid=grid(8192), stream=stream0)
        del primals_521
        del primals_525
        del primals_530
        # Topologically Sorted Source Nodes: [stage3_2_relu_cbrs_1_0_0], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_531, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf257 = buf256; del buf256  # reuse
        buf258 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_relu_cbrs_1_0_0, stage3_2_relu_cbrs_1_0_1, stage3_2_relu_cbrs_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf257, primals_532, primals_533, primals_534, primals_535, primals_536, buf258, 8192, grid=grid(8192), stream=stream0)
        del primals_532
        del primals_536
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_0_0_0], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf224, primals_537, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf259, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf260 = buf259; del buf259  # reuse
        buf261 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_0_0_0, stage3_2_fuse_layers_2_0_0_0_1, stage3_2_fuse_layers_2_0_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf260, primals_538, primals_539, primals_540, primals_541, primals_542, buf261, 4096, grid=grid(4096), stream=stream0)
        del primals_538
        del primals_542
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_0_0_3], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_543, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf263 = buf262; del buf262  # reuse
        buf264 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_0_0_3, stage3_2_fuse_layers_2_0_0_1, stage3_2_fuse_layers_2_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf263, primals_544, primals_545, primals_546, primals_547, primals_548, buf264, 4096, grid=grid(4096), stream=stream0)
        del primals_544
        del primals_548
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_1_0_0], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_549, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf265, (4, 16, 4, 4), (256, 16, 4, 1))
        buf266 = buf265; del buf265  # reuse
        buf267 = empty_strided_cuda((4, 16, 4, 4), (256, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_1_0_0, stage3_2_fuse_layers_2_0_1_0_1, stage3_2_fuse_layers_2_0_1_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf266, primals_550, primals_551, primals_552, primals_553, primals_554, buf267, 1024, grid=grid(1024), stream=stream0)
        del primals_550
        del primals_554
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_1_0_3], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_555, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 56, 4, 4), (896, 16, 4, 1))
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_1_0_0_0], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf233, primals_561, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf270, (4, 32, 4, 4), (512, 16, 4, 1))
        buf271 = buf270; del buf270  # reuse
        buf272 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_1_0_0_0, stage3_2_fuse_layers_2_1_0_0_1, stage3_2_fuse_layers_2_1_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12.run(buf271, primals_562, primals_563, primals_564, primals_565, primals_566, buf272, 2048, grid=grid(2048), stream=stream0)
        del primals_562
        del primals_566
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_1_0_0_3], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_567, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 56, 4, 4), (896, 16, 4, 1))
        buf269 = buf268; del buf268  # reuse
        buf274 = buf273; del buf273  # reuse
        buf275 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [stage3_2_fuse_layers_2_0_1_0_3, stage3_2_fuse_layers_2_0_1_1, stage3_2_fuse_layers_2_1_0_0_3, stage3_2_fuse_layers_2_1_0_1, add_33, add_34], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_20.run(buf269, buf274, buf276, primals_556, primals_568, primals_557, primals_558, primals_559, primals_560, primals_569, primals_570, primals_571, primals_572, buf242, 3584, grid=grid(3584), stream=stream0)
        del primals_556
        del primals_560
        del primals_568
        del primals_572
        # Topologically Sorted Source Nodes: [stage3_2_relu_cbrs_2_0_0], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_573, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 56, 4, 4), (896, 16, 4, 1))
        buf278 = buf277; del buf277  # reuse
        buf279 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_2_relu_cbrs_2_0_0, stage3_2_relu_cbrs_2_0_1, stage3_2_relu_cbrs_2_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf278, primals_574, primals_575, primals_576, primals_577, primals_578, buf279, 3584, grid=grid(3584), stream=stream0)
        del primals_574
        del primals_578
        # Topologically Sorted Source Nodes: [stage3_3_branches_0_0_conv1_0], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf248, primals_579, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf280, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf281 = buf280; del buf280  # reuse
        buf282 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_0_0_conv1_0, stage3_3_branches_0_0_conv1_1, stage3_3_branches_0_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf281, primals_580, primals_581, primals_582, primals_583, primals_584, buf282, 16384, grid=grid(16384), stream=stream0)
        del primals_580
        del primals_584
        # Topologically Sorted Source Nodes: [stage3_3_branches_0_0_conv1_3], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_585, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf284 = buf283; del buf283  # reuse
        buf285 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_0_0_conv1_3, stage3_3_branches_0_0_conv1_4, add_35], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_7.run(buf284, primals_586, primals_587, primals_588, primals_589, primals_590, buf248, buf285, 16384, grid=grid(16384), stream=stream0)
        del primals_586
        del primals_590
        # Topologically Sorted Source Nodes: [stage3_3_branches_0_1_0], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_591, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [stage3_3_branches_1_0_conv1_0], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf258, primals_597, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf288, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf289 = buf288; del buf288  # reuse
        buf290 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_1_0_conv1_0, stage3_3_branches_1_0_conv1_1, stage3_3_branches_1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf289, primals_598, primals_599, primals_600, primals_601, primals_602, buf290, 8192, grid=grid(8192), stream=stream0)
        del primals_598
        del primals_602
        # Topologically Sorted Source Nodes: [stage3_3_branches_1_0_conv1_3], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_603, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf292 = buf291; del buf291  # reuse
        buf293 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_1_0_conv1_3, stage3_3_branches_1_0_conv1_4, add_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf292, primals_604, primals_605, primals_606, primals_607, primals_608, buf258, buf293, 8192, grid=grid(8192), stream=stream0)
        del primals_604
        del primals_608
        # Topologically Sorted Source Nodes: [stage3_3_branches_1_1_0], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_609, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf295 = buf294; del buf294  # reuse
        buf296 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_1_1_0, stage3_3_branches_1_1_1, stage3_3_branches_1_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf295, primals_610, primals_611, primals_612, primals_613, primals_614, buf296, 8192, grid=grid(8192), stream=stream0)
        del primals_610
        del primals_614
        # Topologically Sorted Source Nodes: [stage3_3_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf296, primals_633, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 16, 8, 8), (1024, 64, 8, 1))
        # Topologically Sorted Source Nodes: [stage3_3_branches_2_0_conv1_0], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf279, primals_615, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf297, (4, 56, 4, 4), (896, 16, 4, 1))
        buf298 = buf297; del buf297  # reuse
        buf299 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_2_0_conv1_0, stage3_3_branches_2_0_conv1_1, stage3_3_branches_2_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf298, primals_616, primals_617, primals_618, primals_619, primals_620, buf299, 3584, grid=grid(3584), stream=stream0)
        del primals_616
        del primals_620
        # Topologically Sorted Source Nodes: [stage3_3_branches_2_0_conv1_3], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_621, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 56, 4, 4), (896, 16, 4, 1))
        buf301 = buf300; del buf300  # reuse
        buf302 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_2_0_conv1_3, stage3_3_branches_2_0_conv1_4, add_37], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14.run(buf301, primals_622, primals_623, primals_624, primals_625, primals_626, buf279, buf302, 3584, grid=grid(3584), stream=stream0)
        del primals_622
        del primals_626
        # Topologically Sorted Source Nodes: [stage3_3_branches_2_1_0], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, primals_627, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (4, 56, 4, 4), (896, 16, 4, 1))
        buf304 = buf303; del buf303  # reuse
        buf305 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_branches_2_1_0, stage3_3_branches_2_1_1, stage3_3_branches_2_1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf304, primals_628, primals_629, primals_630, primals_631, primals_632, buf305, 3584, grid=grid(3584), stream=stream0)
        del primals_628
        del primals_632
        # Topologically Sorted Source Nodes: [stage3_3_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf305, primals_638, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 16, 4, 4), (256, 16, 4, 1))
        buf287 = buf286; del buf286  # reuse
        buf307 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf309 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [stage3_3_branches_0_1_0, stage3_3_branches_0_1_1, stage3_3_branches_0_1_2, stage3_3_fuse_layers_0_1_1, stage3_3_fuse_layers_0_1_2, add_38, stage3_3_fuse_layers_0_2_1, stage3_3_fuse_layers_0_2_2, add_39], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_relu_21.run(buf287, buf309, primals_592, primals_593, primals_594, primals_595, primals_596, buf66, buf306, primals_634, primals_635, primals_636, primals_637, buf115, buf308, primals_639, primals_640, primals_641, primals_642, 16384, grid=grid(16384), stream=stream0)
        del primals_592
        del primals_637
        del primals_642
        # Topologically Sorted Source Nodes: [stage3_3_relu_cbrs_0_0_0], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_643, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf311 = buf310; del buf310  # reuse
        buf312 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stage3_3_relu_cbrs_0_0_0, stage3_3_relu_cbrs_0_0_1, stage3_3_relu_cbrs_0_0_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(buf311, primals_644, primals_645, primals_646, primals_647, primals_648, buf312, 16384, grid=grid(16384), stream=stream0)
        del primals_644
        del primals_648
        # Topologically Sorted Source Nodes: [final_layers_0], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_649, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 46, 16, 16), (11776, 256, 16, 1))
        buf314 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [final_layers_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf314, primals_650, 47104, grid=grid(47104), stream=stream0)
        del primals_650
        buf315 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [deconv_layers_0_0_0], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_23.run(buf315, 32, grid=grid(32), stream=stream0)
        buf316 = empty_strided_cuda((4, 62, 32, 32), (63488, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1, deconv_layers_0_0_0], Original ATen: [aten.cat, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_cat_24.run(buf315, buf312, buf314, buf316, 253952, grid=grid(253952), stream=stream0)
        # Topologically Sorted Source Nodes: [deconv_layers_0_0_1], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_651, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf318 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [deconv_layers_0_0_2, deconv_layers_0_0_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf317, primals_652, primals_653, primals_654, primals_655, buf318, 65536, grid=grid(65536), stream=stream0)
        del primals_655
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv1_0], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_656, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf319, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf320 = buf319; del buf319  # reuse
        buf321 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv1_0, deconv_layers_0_1_0_conv1_1, deconv_layers_0_1_0_conv1_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(buf320, primals_657, primals_658, primals_659, primals_660, primals_661, buf321, 65536, grid=grid(65536), stream=stream0)
        del primals_657
        del primals_661
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv1_3], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_662, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf323 = buf322; del buf322  # reuse
        buf324 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv1_3, deconv_layers_0_1_0_conv1_4, deconv_layers_0_1_0_relu1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(buf323, primals_663, primals_664, primals_665, primals_666, primals_667, buf324, 65536, grid=grid(65536), stream=stream0)
        del primals_663
        del primals_667
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv2_0], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, primals_668, stride=(1, 1), padding=(3, 3), dilation=(3, 3), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf325, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf326 = buf325; del buf325  # reuse
        buf327 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv2_0, deconv_layers_0_1_0_conv2_1, deconv_layers_0_1_0_conv2_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(buf326, primals_669, primals_670, primals_671, primals_672, primals_673, buf327, 65536, grid=grid(65536), stream=stream0)
        del primals_669
        del primals_673
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv2_3], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_674, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 16, 32, 32), (16384, 1024, 32, 1))
        buf329 = buf328; del buf328  # reuse
        buf330 = empty_strided_cuda((4, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [deconv_layers_0_1_0_conv2_3, deconv_layers_0_1_0_conv2_4, add_40], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27.run(buf329, primals_675, primals_676, primals_677, primals_678, primals_679, buf318, buf330, 65536, grid=grid(65536), stream=stream0)
        del primals_675
        del primals_679
        # Topologically Sorted Source Nodes: [final_layers_1], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_680, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 23, 32, 32), (23552, 1024, 32, 1))
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [final_layers_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf332, primals_681, 94208, grid=grid(94208), stream=stream0)
        del primals_681
    return (buf314, buf332, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_9, primals_10, primals_11, primals_13, primals_15, primals_16, primals_17, primals_19, primals_21, primals_22, primals_23, primals_25, primals_27, primals_28, primals_29, primals_31, primals_33, primals_34, primals_35, primals_37, primals_39, primals_40, primals_41, primals_43, primals_45, primals_46, primals_47, primals_49, primals_51, primals_52, primals_53, primals_55, primals_57, primals_58, primals_59, primals_61, primals_63, primals_64, primals_65, primals_67, primals_69, primals_70, primals_71, primals_73, primals_75, primals_76, primals_77, primals_79, primals_81, primals_82, primals_83, primals_85, primals_87, primals_88, primals_89, primals_91, primals_93, primals_94, primals_95, primals_97, primals_99, primals_100, primals_101, primals_103, primals_105, primals_106, primals_107, primals_109, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_121, primals_123, primals_124, primals_125, primals_127, primals_129, primals_130, primals_131, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_144, primals_146, primals_147, primals_148, primals_150, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_186, primals_188, primals_189, primals_190, primals_192, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_204, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_216, primals_218, primals_219, primals_220, primals_222, primals_224, primals_225, primals_226, primals_228, primals_229, primals_230, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_244, primals_246, primals_247, primals_248, primals_250, primals_252, primals_253, primals_254, primals_256, primals_257, primals_258, primals_259, primals_261, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_273, primals_275, primals_276, primals_277, primals_279, primals_281, primals_282, primals_283, primals_285, primals_287, primals_288, primals_289, primals_291, primals_293, primals_294, primals_295, primals_297, primals_299, primals_300, primals_301, primals_303, primals_305, primals_306, primals_307, primals_309, primals_311, primals_312, primals_313, primals_315, primals_317, primals_318, primals_319, primals_321, primals_323, primals_324, primals_325, primals_327, primals_329, primals_330, primals_331, primals_333, primals_335, primals_336, primals_337, primals_339, primals_341, primals_342, primals_343, primals_345, primals_347, primals_348, primals_349, primals_351, primals_353, primals_354, primals_355, primals_357, primals_359, primals_360, primals_361, primals_363, primals_364, primals_365, primals_366, primals_368, primals_369, primals_370, primals_371, primals_373, primals_375, primals_376, primals_377, primals_379, primals_381, primals_382, primals_383, primals_385, primals_387, primals_388, primals_389, primals_391, primals_392, primals_393, primals_394, primals_396, primals_398, primals_399, primals_400, primals_402, primals_404, primals_405, primals_406, primals_408, primals_410, primals_411, primals_412, primals_414, primals_416, primals_417, primals_418, primals_420, primals_422, primals_423, primals_424, primals_426, primals_428, primals_429, primals_430, primals_432, primals_434, primals_435, primals_436, primals_438, primals_440, primals_441, primals_442, primals_444, primals_446, primals_447, primals_448, primals_450, primals_452, primals_453, primals_454, primals_456, primals_458, primals_459, primals_460, primals_462, primals_464, primals_465, primals_466, primals_468, primals_470, primals_471, primals_472, primals_474, primals_476, primals_477, primals_478, primals_480, primals_482, primals_483, primals_484, primals_486, primals_488, primals_489, primals_490, primals_492, primals_494, primals_495, primals_496, primals_498, primals_499, primals_500, primals_501, primals_503, primals_504, primals_505, primals_506, primals_508, primals_510, primals_511, primals_512, primals_514, primals_516, primals_517, primals_518, primals_520, primals_522, primals_523, primals_524, primals_526, primals_527, primals_528, primals_529, primals_531, primals_533, primals_534, primals_535, primals_537, primals_539, primals_540, primals_541, primals_543, primals_545, primals_546, primals_547, primals_549, primals_551, primals_552, primals_553, primals_555, primals_557, primals_558, primals_559, primals_561, primals_563, primals_564, primals_565, primals_567, primals_569, primals_570, primals_571, primals_573, primals_575, primals_576, primals_577, primals_579, primals_581, primals_582, primals_583, primals_585, primals_587, primals_588, primals_589, primals_591, primals_593, primals_594, primals_595, primals_596, primals_597, primals_599, primals_600, primals_601, primals_603, primals_605, primals_606, primals_607, primals_609, primals_611, primals_612, primals_613, primals_615, primals_617, primals_618, primals_619, primals_621, primals_623, primals_624, primals_625, primals_627, primals_629, primals_630, primals_631, primals_633, primals_634, primals_635, primals_636, primals_638, primals_639, primals_640, primals_641, primals_643, primals_645, primals_646, primals_647, primals_649, primals_651, primals_652, primals_653, primals_654, primals_656, primals_658, primals_659, primals_660, primals_662, primals_664, primals_665, primals_666, primals_668, primals_670, primals_671, primals_672, primals_674, primals_676, primals_677, primals_678, primals_680, buf0, buf1, buf3, buf4, buf6, buf7, buf9, buf10, buf12, buf14, buf15, buf17, buf19, buf21, buf22, buf24, buf25, buf27, buf28, buf30, buf31, buf33, buf34, buf36, buf37, buf39, buf40, buf42, buf43, buf45, buf46, buf48, buf49, buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf63, buf64, buf65, buf66, buf67, buf69, buf70, buf72, buf73, buf75, buf76, buf78, buf79, buf81, buf82, buf84, buf85, buf87, buf88, buf90, buf91, buf93, buf94, buf96, buf97, buf99, buf100, buf102, buf103, buf105, buf106, buf108, buf109, buf111, buf112, buf113, buf114, buf115, buf116, buf118, buf119, buf121, buf122, buf124, buf125, buf126, buf127, buf129, buf130, buf132, buf133, buf135, buf136, buf138, buf139, buf141, buf143, buf144, buf146, buf148, buf150, buf151, buf153, buf154, buf156, buf157, buf159, buf160, buf162, buf163, buf165, buf166, buf168, buf169, buf171, buf172, buf174, buf175, buf177, buf178, buf179, buf180, buf181, buf183, buf184, buf186, buf187, buf189, buf190, buf191, buf193, buf194, buf196, buf197, buf199, buf200, buf202, buf203, buf205, buf207, buf208, buf210, buf212, buf214, buf215, buf217, buf218, buf220, buf221, buf223, buf224, buf226, buf227, buf229, buf230, buf232, buf233, buf235, buf236, buf238, buf239, buf241, buf242, buf243, buf244, buf245, buf247, buf248, buf250, buf251, buf253, buf254, buf255, buf257, buf258, buf260, buf261, buf263, buf264, buf266, buf267, buf269, buf271, buf272, buf274, buf276, buf278, buf279, buf281, buf282, buf284, buf285, buf287, buf289, buf290, buf292, buf293, buf295, buf296, buf298, buf299, buf301, buf302, buf304, buf305, buf306, buf308, buf309, buf311, buf312, buf315, buf316, buf317, buf318, buf320, buf321, buf323, buf324, buf326, buf327, buf329, buf330, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((56, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((16, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((32, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((56, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((56, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((16, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((32, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((56, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((56, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((16, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((32, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((56, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((56, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((16, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((46, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((46, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((16, 62, 1, 1), (62, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((23, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((23, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
