# AOT ID: ['21_forward']
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


# kernel path: inductor_cache/z3/cz3crgbvsv4gzr6vwfosmhdzympe4r2fphifxocc5mq3wn7nndqt.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_8), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_11), kwargs = {})
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
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32768) % 64)
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


# kernel path: inductor_cache/x7/cx7l6ozqecnv6uj7rtsy6zhcqjwlazhqv7clil35wsk7hppmo4lv.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_3, mul_4, mul_5, sub_1
#   out_2 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_14), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_17), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_20), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_23), kwargs = {})
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
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
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


# kernel path: inductor_cache/hr/chrufz6iwk6iascfavwkwrptmxmdl7zlfzi2qxehcpuih42zl4og.py
# Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_4 => add_5, mul_7, mul_8, sub_2
#   out_5 => add_6
#   out_6 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_26), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_29), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_32), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_35), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %getitem), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/n4/cn4ckisnhcdft35flrcyo4ufwur2mqc46l5ozml5mbv2uhjsxo2a.py
# Topologically Sorted Source Nodes: [out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_8 => add_8, mul_10, mul_11, sub_3
#   out_9 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_38), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_41), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_44), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_47), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 128)
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


# kernel path: inductor_cache/ef/cefr6hyzcw546gna7aio5nxh2andpurwqp6b4n4wzlwokhd7sgg7.py
# Topologically Sorted Source Nodes: [out_11, input_2, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_12, mul_16, mul_17, sub_5
#   out_11 => add_10, mul_13, mul_14, sub_4
#   out_12 => add_13
#   out_13 => relu_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_50), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_53), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_56), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_59), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_62), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_65), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_68), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_71), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %add_12), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/nw/cnwz3qgydodgmfoohgpzc22lwwlubhlsxaxegskujj3b4ehldbr7.py
# Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_15 => add_15, mul_19, mul_20, sub_6
#   out_16 => relu_5
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_74), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_77), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_80), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_83), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 256)
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


# kernel path: inductor_cache/ax/cax5qqoh7w4o4cozecxymwuyjkl5b5w5e2zkfdxl4gnnj4gfksiv.py
# Topologically Sorted Source Nodes: [out_18, input_4, out_19, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_19, mul_25, mul_26, sub_8
#   out_18 => add_17, mul_22, mul_23, sub_7
#   out_19 => add_20
#   out_20 => relu_6
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_86), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_89), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_92), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_95), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_98), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_101), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_104), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_107), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %add_19), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_20,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/6b/c6bn5cwlqgzey6whbva22aqrpp64w4q66vdre3dsihwitmxcgm2z.py
# Topologically Sorted Source Nodes: [out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_22 => add_22, mul_28, mul_29, sub_9
#   out_23 => relu_7
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_110), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_113), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_116), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_119), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_22,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 512)
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


# kernel path: inductor_cache/na/cnawilhgaq2o5kfs5wstftybm6lfrouz4lug7g6ms7kf6qkdpjw3.py
# Topologically Sorted Source Nodes: [out_25, input_6, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_26, mul_34, mul_35, sub_11
#   out_25 => add_24, mul_31, mul_32, sub_10
#   out_26 => add_27
#   out_27 => relu_8
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_122), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_125), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_128), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_131), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_134), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_137), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_140), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_143), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %add_26), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/wl/cwlzrfk7l66kfaxvh6duuyogq5wshxam5njd2n5lg2fthx2nq3md.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_7 => convolution_12
#   input_8 => add_29, mul_37, mul_38, sub_12
#   input_9 => relu_9
# Graph fragment:
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_62, %primals_63, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_146), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_149), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_152), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_155), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 256)
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


# kernel path: inductor_cache/p7/cp7fymlqbprcuprg5ucid5tkas3qhngtajfrcf7px4kg5lshvw2q.py
# Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => convolution_13
#   input_11 => add_31, mul_40, mul_41, sub_13
#   input_12 => relu_10
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_68, %primals_69, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_158), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_161), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_164), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_167), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32768) % 256)
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


# kernel path: inductor_cache/54/c543glgfncwhqvbsuwqcfcofwvgu3bgehn5vqfovz6fpzbftnozp.py
# Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_13 => convolution_14
#   input_14 => add_33, mul_43, mul_44, sub_14
#   input_15 => relu_11
# Graph fragment:
#   %convolution_14 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %primals_74, %primals_75, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_170), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_173), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_176), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_179), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 256)
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


# kernel path: inductor_cache/5i/c5i45gpxvqa5bytzux5l5kpfsiipau3fbntukhnkch4p65xtgida.py
# Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_17 => add_35, mul_46, mul_47, sub_15
#   input_18 => relu_12
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_182), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_185), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_188), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_191), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 128)
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7, 7), (1029, 343, 49, 7, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64, 64), (786432, 262144, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (128, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (256, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (512, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_62, (512, 256, 2, 2, 2), (2048, 8, 4, 2, 1))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, 256, 2, 2, 2), (2048, 8, 4, 2, 1))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, 256, 2, 2, 2), (2048, 8, 4, 2, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (128, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (10, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2, 2), padding=(3, 3, 3), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 8388608, grid=grid(8388608), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool3d_with_indices]
        buf2 = torch.ops.aten.max_pool3d_with_indices.default(buf1, [3, 3, 3], [2, 2, 2], [1, 1, 1])
        buf3 = buf2[0]
        buf4 = buf2[1]
        del buf2
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, primals_7, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf6 = empty_strided_cuda((4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf5, primals_8, primals_9, primals_10, primals_11, buf6, 1048576, grid=grid(1048576), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_12, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf8 = empty_strided_cuda((4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2.run(buf7, primals_13, primals_14, primals_15, primals_16, buf3, buf8, 1048576, grid=grid(1048576), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_17, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf10 = empty_strided_cuda((4, 128, 8, 8, 8), (65536, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf9, primals_18, primals_19, primals_20, primals_21, buf10, 262144, grid=grid(262144), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_22, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf8, primals_27, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf13 = empty_strided_cuda((4, 128, 8, 8, 8), (65536, 512, 64, 8, 1), torch.float32)
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [out_11, input_2, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf14, buf11, primals_23, primals_24, primals_25, primals_26, buf12, primals_28, primals_29, primals_30, primals_31, 262144, grid=grid(262144), stream=stream0)
        del primals_26
        del primals_31
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_32, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf16 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf15, primals_33, primals_34, primals_35, primals_36, buf16, 524288, grid=grid(524288), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_37, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf14, primals_42, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf19 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [out_18, input_4, out_19, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf20, buf17, primals_38, primals_39, primals_40, primals_41, buf18, primals_43, primals_44, primals_45, primals_46, 524288, grid=grid(524288), stream=stream0)
        del primals_41
        del primals_46
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_47, stride=(1, 1, 1), padding=(4, 4, 4), dilation=(4, 4, 4), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        buf22 = empty_strided_cuda((4, 512, 8, 8, 8), (262144, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf21, primals_48, primals_49, primals_50, primals_51, buf22, 1048576, grid=grid(1048576), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_52, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf20, primals_57, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 512, 8, 8, 8), (262144, 512, 64, 8, 1))
        buf25 = empty_strided_cuda((4, 512, 8, 8, 8), (262144, 512, 64, 8, 1), torch.float32)
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [out_25, input_6, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf26, buf23, primals_53, primals_54, primals_55, primals_56, buf24, primals_58, primals_59, primals_60, primals_61, 1048576, grid=grid(1048576), stream=stream0)
        del primals_56
        del primals_61
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_62, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1))
        buf28 = buf27; del buf27  # reuse
        buf29 = empty_strided_cuda((4, 256, 16, 16, 16), (1048576, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf28, primals_63, primals_64, primals_65, primals_66, primals_67, buf29, 4194304, grid=grid(4194304), stream=stream0)
        del primals_63
        del primals_67
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_68, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 256, 32, 32, 32), (8388608, 32768, 1024, 32, 1))
        buf31 = buf30; del buf30  # reuse
        buf32 = empty_strided_cuda((4, 256, 32, 32, 32), (8388608, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf31, primals_69, primals_70, primals_71, primals_72, primals_73, buf32, 33554432, grid=grid(33554432), stream=stream0)
        del primals_69
        del primals_73
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_74, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 256, 64, 64, 64), (67108864, 262144, 4096, 64, 1))
        buf34 = buf33; del buf33  # reuse
        buf35 = empty_strided_cuda((4, 256, 64, 64, 64), (67108864, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf34, primals_75, primals_76, primals_77, primals_78, primals_79, buf35, 268435456, grid=grid(268435456), stream=stream0)
        del primals_75
        del primals_79
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_80, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 128, 64, 64, 64), (33554432, 262144, 4096, 64, 1))
        buf37 = empty_strided_cuda((4, 128, 64, 64, 64), (33554432, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf36, primals_81, primals_82, primals_83, primals_84, buf37, 134217728, grid=grid(134217728), stream=stream0)
        del primals_84
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_85, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 10, 64, 64, 64), (2621440, 262144, 4096, 64, 1))
    return (buf38, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_64, primals_65, primals_66, primals_68, primals_70, primals_71, primals_72, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_83, primals_85, buf0, buf1, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf18, buf20, buf21, buf22, buf23, buf24, buf26, buf28, buf29, buf31, buf32, buf34, buf35, buf36, buf37, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7, 7), (1029, 343, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64, 64), (786432, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, 256, 2, 2, 2), (2048, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 256, 2, 2, 2), (2048, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 256, 2, 2, 2), (2048, 8, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((10, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
