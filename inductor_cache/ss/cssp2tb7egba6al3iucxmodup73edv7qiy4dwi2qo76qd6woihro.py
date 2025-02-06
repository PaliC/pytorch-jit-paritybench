# AOT ID: ['105_forward']
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


# kernel path: inductor_cache/in/cinbtjb3wzgcalqsafqwntomsd7unebj7ez5f2dbwzu23zb7yri6.py
# Topologically Sorted Source Nodes: [batch_norm, silu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   silu => mul_3, sigmoid
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b7/cb74bzsajzwrgpig7hmwazhk2hyisgdav4utq2amlbdkbp6uhu5o.py
# Topologically Sorted Source Nodes: [batch_norm_1, silu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_5, mul_6, sub_1
#   silu_1 => mul_7, sigmoid_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr73chiqx2ddg5g3lq5yrczpesbp3w5stypfx3gvzly7ulklc2y7.py
# Topologically Sorted Source Nodes: [batch_norm_3, silu_3, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
# Source node to ATen node mapping:
#   batch_norm_3 => add_7, mul_13, mul_14, sub_3
#   input_1 => add_8
#   silu_3 => mul_15, sigmoid_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_7,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %sigmoid_3), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
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
    tmp17 = tl.sigmoid(tmp15)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp16 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p5/cp5tskbvhf6dstshzzdqeypi7iqem6vlpge6gvuf6ph6vgim4syw.py
# Topologically Sorted Source Nodes: [batch_norm_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_4 => add_10, mul_17, mul_18, sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/qa/cqa45mnnnqzcvufmhv4hwi7b3gis3ehapbq4fdeolbpnuqjihkfy.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_19, %mul_23], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 4, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (x0 + 16*((-2) + x1) + 32*x2), tmp10 & xmask, other=0.0)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6d/c6ddtocrltjxsvpe5yghv7rfeersboxli4tmazxrafhtyiabf6rv.py
# Topologically Sorted Source Nodes: [batch_norm_6, silu_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   batch_norm_6 => add_14, mul_25, mul_26, sub_6
#   silu_6 => mul_27, sigmoid_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_53), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_55), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_14,), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %sigmoid_6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36 = args
    args.clear()
    assert_size_stride(primals_1, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (2, ), (1, ))
    assert_size_stride(primals_4, (2, ), (1, ))
    assert_size_stride(primals_5, (2, ), (1, ))
    assert_size_stride(primals_6, (2, ), (1, ))
    assert_size_stride(primals_7, (1, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_8, (1, ), (1, ))
    assert_size_stride(primals_9, (1, ), (1, ))
    assert_size_stride(primals_10, (1, ), (1, ))
    assert_size_stride(primals_11, (1, ), (1, ))
    assert_size_stride(primals_12, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_13, (1, ), (1, ))
    assert_size_stride(primals_14, (1, ), (1, ))
    assert_size_stride(primals_15, (1, ), (1, ))
    assert_size_stride(primals_16, (1, ), (1, ))
    assert_size_stride(primals_17, (2, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_18, (2, ), (1, ))
    assert_size_stride(primals_19, (2, ), (1, ))
    assert_size_stride(primals_20, (2, ), (1, ))
    assert_size_stride(primals_21, (2, ), (1, ))
    assert_size_stride(primals_22, (2, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(primals_23, (2, ), (1, ))
    assert_size_stride(primals_24, (2, ), (1, ))
    assert_size_stride(primals_25, (2, ), (1, ))
    assert_size_stride(primals_26, (2, ), (1, ))
    assert_size_stride(primals_27, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_28, (2, ), (1, ))
    assert_size_stride(primals_29, (2, ), (1, ))
    assert_size_stride(primals_30, (2, ), (1, ))
    assert_size_stride(primals_31, (2, ), (1, ))
    assert_size_stride(primals_32, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 2, 4, 4), (32, 16, 4, 1))
        buf1 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [batch_norm, silu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_0.run(buf2, buf0, primals_3, primals_4, primals_5, primals_6, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 1, 4, 4), (16, 16, 4, 1))
        buf4 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf5 = reinterpret_tensor(buf4, (4, 1, 4, 4), (16, 16, 4, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_1, silu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_1.run(buf5, buf3, primals_8, primals_9, primals_10, primals_11, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 1, 4, 4), (16, 16, 4, 1))
        buf7 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf8 = reinterpret_tensor(buf7, (4, 1, 4, 4), (16, 16, 4, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_2, silu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_1.run(buf8, buf6, primals_13, primals_14, primals_15, primals_16, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 2, 4, 4), (32, 16, 4, 1))
        buf10 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_3, silu_3, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_2.run(buf11, buf9, primals_18, primals_19, primals_20, primals_21, buf2, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 2, 4, 4), (32, 16, 4, 1))
        buf13 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf12, primals_23, primals_24, primals_25, primals_26, buf13, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(primals_2, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 2, 4, 4), (32, 16, 4, 1))
        buf15 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf14, primals_28, primals_29, primals_30, primals_31, buf15, 128, grid=grid(128), stream=stream0)
        buf16 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf13, buf15, buf16, 256, grid=grid(256), stream=stream0)
        del buf13
        del buf15
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 4, 4, 4), (64, 16, 4, 1))
        buf18 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_6, silu_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_silu_5.run(buf19, buf17, primals_33, primals_34, primals_35, primals_36, 256, grid=grid(256), stream=stream0)
    return (buf19, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf16, buf17, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2, 2, 1, 1), (2, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
