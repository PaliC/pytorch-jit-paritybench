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


# kernel path: inductor_cache/ld/cldiju5y7egebjdayc2znize7jv6efgcc5wyr6rdxajhqez33ovl.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => clamp_max, clamp_min
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 9)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/35/c35d3p5bsrk6vssibmtxtytrq63af3h652h5fh6ue63mtvp2jklx.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => clamp_max_1, clamp_min_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 9)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/sx/csxj6qgdm6ifndgsq5nb6knaieoe6w5gqdrdix5nvhnqqs2tqhyc.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_7, mul_8, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/p3/cp3mi6pwdgettq26dj7oajcwkzzktz7nvt4irksjec6gfsuljlnl.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_10 => add_7, mul_10, mul_11, sub_3
#   input_11 => clamp_max_2, clamp_min_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 96)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/wd/cwdb2tzvf7nehkddqqmt6alpqwjr3ltu3amspljwa3kinjytwdpl.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_16 => add_11, mul_16, mul_17, sub_5
#   input_17 => add_12
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %add_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ah/cahcu53ti5eszaxq5qfe6z4wxaenk6atp72tl2fvtxxhxp47b5oz.py
# Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_28 => add_21, mul_28, mul_29, sub_9
#   input_29 => clamp_max_6, clamp_min_6
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_21, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/rv/crvzpwkp4ec4bvv3ng7ebehlewkviwpm6rpkbffzem2yemcaxfuc.py
# Topologically Sorted Source Nodes: [input_37, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_37 => add_27, mul_37, mul_38, sub_12
#   out => add_28
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %add_28 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %add_19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5igem7rj6kshb4frpp637jfaosfkxxnooo2mprmhgjnqrnfdyne.py
# Topologically Sorted Source Nodes: [input_61, input_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_61 => add_48, mul_64, mul_65, sub_21
#   input_62 => clamp_max_15, clamp_min_15
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_48, 0.0), kwargs = {})
#   %clamp_max_15 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 32)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/az/caz46p7gqe6exxcorv2kbigu3hnmzjdcrl42vsalrqge7w57rjxm.py
# Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_64 => add_50, mul_67, mul_68, sub_22
#   input_65 => clamp_max_16, clamp_min_16
# Graph fragment:
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_50, 0.0), kwargs = {})
#   %clamp_max_16 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/yn/cyndduc7kmvngibm74rgfzjsznma2pzxhotyysd7t6qsgpn4qhbe.py
# Topologically Sorted Source Nodes: [input_70, input_72, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_70 => add_54, mul_73, mul_74, sub_24
#   input_72 => add_56, mul_76, mul_77, sub_25
#   out_3 => add_57
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %add_57 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %add_56), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
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
    tl.store(out_ptr0 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/2g/c2g4ufpffqfongwdan5hjfxvb2vy7kqr5qehtd3b7hbcskna5ogb.py
# Topologically Sorted Source Nodes: [input_83, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_83 => add_65, mul_88, mul_89, sub_29
#   out_4 => add_66
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_237), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_239), kwargs = {})
#   %add_66 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %add_57), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cik27fioiyvngg7ocpwqjafqqh3jxt2d56i6hfv4kchk2wydcqny.py
# Topologically Sorted Source Nodes: [input_242, input_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_242 => add_196, mul_262, mul_263, sub_87
#   input_243 => clamp_max_64, clamp_min_64
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %clamp_min_64 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_196, 0.0), kwargs = {})
#   %clamp_max_64 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_64, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/pw/cpweiw3x2ai34i2is3t4pacxrc3xj47vpl45wufsizutz3p4kyy7.py
# Topologically Sorted Source Nodes: [input_248, input_250, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_248 => add_200, mul_268, mul_269, sub_89
#   input_250 => add_202, mul_271, mul_272, sub_90
#   out_19 => add_203
# Graph fragment:
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_89, %unsqueeze_713), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %unsqueeze_717), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %unsqueeze_719), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_721), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_725), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_727), kwargs = {})
#   %add_203 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_200, %add_202), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/6t/c6tbsgkggiqbkxsvtbjge4ky63ob57e7yiqouthr2kh2duicgws6.py
# Topologically Sorted Source Nodes: [input_261, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_261 => add_211, mul_283, mul_284, sub_94
#   out_20 => add_212
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_757), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_759), kwargs = {})
#   %add_212 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %add_203), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/qe/cqe3zhh5uswgkvvmddhrxuxwlnkkfqmhzzw5ln3bwq77ge4n4qi7.py
# Topologically Sorted Source Nodes: [feature_volume], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   feature_volume => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_192, %add_221, %add_248], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 320)
    x0 = (xindex % 256)
    x2 = xindex // 81920
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 16384*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-64) + x1) + 32768*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 256*((-192) + x1) + 32768*x2), tmp11, other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-192) + x1), tmp11, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 - tmp15
    tmp17 = tl.load(in_ptr4 + ((-192) + x1), tmp11, eviction_policy='evict_last', other=0.0)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tl.full([1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp16 * tmp24
    tmp26 = tl.load(in_ptr5 + ((-192) + x1), tmp11, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr6 + ((-192) + x1), tmp11, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.load(in_ptr7 + (x0 + 256*((-192) + x1) + 32768*x2), tmp11, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp11, tmp31, tmp32)
    tmp34 = tl.where(tmp9, tmp10, tmp33)
    tmp35 = tl.where(tmp4, tmp5, tmp34)
    tl.store(out_ptr0 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/qq/cqqmctc7nyzjkkr6i677rlo3huqvxdb5bd6e4ip75tz2gxkkoeov.py
# Topologically Sorted Source Nodes: [input_307, input_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_307 => add_250, mul_334, mul_335, sub_111
#   input_308 => clamp_max_81, clamp_min_81
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_1), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_3), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_5), kwargs = {})
#   %add_250 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_7), kwargs = {})
#   %clamp_min_81 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_250, 0.0), kwargs = {})
#   %clamp_max_81 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_81, 6.0), kwargs = {})
#   %le_131 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_250, 0.0), kwargs = {})
#   %ge_124 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_250, 6.0), kwargs = {})
#   %bitwise_or_124 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_131, %ge_124), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 9)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp15 <= tmp16
    tmp21 = tmp15 >= tmp18
    tmp22 = tmp20 | tmp21
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/yu/cyuztxhho4vfljc44oakfekswgedjvtf72tcbvtsz3sjjvrcb7vo.py
# Topologically Sorted Source Nodes: [input_310, input_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_310 => add_252, mul_337, mul_338, sub_112
#   input_311 => clamp_max_82, clamp_min_82
# Graph fragment:
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_9), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_11), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_13), kwargs = {})
#   %add_252 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_15), kwargs = {})
#   %clamp_min_82 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_252, 0.0), kwargs = {})
#   %clamp_max_82 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_82, 6.0), kwargs = {})
#   %le_130 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_252, 0.0), kwargs = {})
#   %ge_123 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_252, 6.0), kwargs = {})
#   %bitwise_or_123 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_130, %ge_123), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 9)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp15 <= tmp16
    tmp21 = tmp15 >= tmp18
    tmp22 = tmp20 | tmp21
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/ru/cruyxad7ejsy57zos2kflupyuoecbxx7zmapf56yijzlhmi7ftgz.py
# Topologically Sorted Source Nodes: [input_315, input_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_315 => add_256, mul_343, mul_344, sub_114
#   input_316 => clamp_max_83, clamp_min_83
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_25), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_27), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_29), kwargs = {})
#   %add_256 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_31), kwargs = {})
#   %clamp_min_83 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_256, 0.0), kwargs = {})
#   %clamp_max_83 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_83, 6.0), kwargs = {})
#   %le_129 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_256, 0.0), kwargs = {})
#   %ge_122 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_256, 6.0), kwargs = {})
#   %bitwise_or_122 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_129, %ge_122), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 96)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp15 <= tmp16
    tmp21 = tmp15 >= tmp18
    tmp22 = tmp20 | tmp21
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/bq/cbq4t7z5r7enxewzxcvbollqmhym52lgapqjnlyaectj6jcfcvnu.py
# Topologically Sorted Source Nodes: [input_333, input_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_333 => add_270, mul_361, mul_362, sub_120
#   input_334 => clamp_max_87, clamp_min_87
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_73), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_75), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_361, %unsqueeze_77), kwargs = {})
#   %add_270 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_362, %unsqueeze_79), kwargs = {})
#   %clamp_min_87 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_270, 0.0), kwargs = {})
#   %clamp_max_87 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_87, 6.0), kwargs = {})
#   %le_125 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_270, 0.0), kwargs = {})
#   %ge_118 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_270, 6.0), kwargs = {})
#   %bitwise_or_118 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_125, %ge_118), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp15 <= tmp16
    tmp21 = tmp15 >= tmp18
    tmp22 = tmp20 | tmp21
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbrklaildyw3ybvz3miyamnmcu23qcvopsjcy7sm627arc27he5a.py
# Topologically Sorted Source Nodes: [input_366, input_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_366 => add_297, mul_397, mul_398, sub_132
#   input_367 => clamp_max_96, clamp_min_96
# Graph fragment:
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_132, %unsqueeze_169), kwargs = {})
#   %mul_397 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_132, %unsqueeze_171), kwargs = {})
#   %mul_398 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_397, %unsqueeze_173), kwargs = {})
#   %add_297 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_398, %unsqueeze_175), kwargs = {})
#   %clamp_min_96 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_297, 0.0), kwargs = {})
#   %clamp_max_96 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_96, 6.0), kwargs = {})
#   %le_116 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_297, 0.0), kwargs = {})
#   %ge_109 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_297, 6.0), kwargs = {})
#   %bitwise_or_109 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_116, %ge_109), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 32)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp15 <= tmp16
    tmp21 = tmp15 >= tmp18
    tmp22 = tmp20 | tmp21
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/k6/ck63c7kdc33nz56ql5nyfpspg46p5wmhppayfpkhxnttyfcct6mo.py
# Topologically Sorted Source Nodes: [input_369, input_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_369 => add_299, mul_400, mul_401, sub_133
#   input_370 => clamp_max_97, clamp_min_97
# Graph fragment:
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_133, %unsqueeze_177), kwargs = {})
#   %mul_400 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %unsqueeze_179), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_400, %unsqueeze_181), kwargs = {})
#   %add_299 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_401, %unsqueeze_183), kwargs = {})
#   %clamp_min_97 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_299, 0.0), kwargs = {})
#   %clamp_max_97 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_97, 6.0), kwargs = {})
#   %le_115 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_299, 0.0), kwargs = {})
#   %ge_108 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_299, 6.0), kwargs = {})
#   %bitwise_or_108 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_115, %ge_108), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp15 <= tmp16
    tmp21 = tmp15 >= tmp18
    tmp22 = tmp20 | tmp21
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/qu/cqul7por4jyn2rgvpsuqxl7yiqxvk6oau4izxoxjgorytylecv4v.py
# Topologically Sorted Source Nodes: [input_547, input_548], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   input_547 => add_445, mul_595, mul_596, sub_198
#   input_548 => clamp_max_145, clamp_min_145
# Graph fragment:
#   %sub_198 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_198, %unsqueeze_697), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_198, %unsqueeze_699), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_595, %unsqueeze_701), kwargs = {})
#   %add_445 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_596, %unsqueeze_703), kwargs = {})
#   %clamp_min_145 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_445, 0.0), kwargs = {})
#   %clamp_max_145 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_145, 6.0), kwargs = {})
#   %le_67 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_445, 0.0), kwargs = {})
#   %ge_60 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_445, 6.0), kwargs = {})
#   %bitwise_or_60 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_67, %ge_60), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp15 <= tmp16
    tmp21 = tmp15 >= tmp18
    tmp22 = tmp20 | tmp21
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/da/cdabaftt3e4bnd3ey2w7mnpdav55qsh3zlxjjolpwrebvvuyhkgx.py
# Topologically Sorted Source Nodes: [volume, cost], Original ATen: [aten.new_zeros, aten.mean]
# Source node to ATen node mapping:
#   cost => mean
#   volume => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 40, 1, 16, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view, [2]), kwargs = {})
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mean, 2, 0), kwargs = {})
triton_per_fused_mean_new_zeros_22 = async_compile.triton('triton_per_fused_mean_new_zeros_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_new_zeros_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_new_zeros_22(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r2 + 2048*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r2 + 2048*x1), None)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = tmp6 == tmp6
    tmp8 = 8.0
    tmp9 = tmp5 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/a7/ca7uz73i2fqtvxg56bbmir6saqkm45cgfve5a7lwhen7mjpmntnb.py
# Topologically Sorted Source Nodes: [input_612, input_613], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_612 => add_499, mul_668, mul_669, sub_222
#   input_613 => clamp_max_162, clamp_min_162
# Graph fragment:
#   %sub_222 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_222, %unsqueeze_1778), kwargs = {})
#   %mul_668 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_222, %unsqueeze_1781), kwargs = {})
#   %mul_669 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_668, %unsqueeze_1784), kwargs = {})
#   %add_499 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_669, %unsqueeze_1787), kwargs = {})
#   %clamp_min_162 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_499, 0.0), kwargs = {})
#   %clamp_max_162 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_162, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 120)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ao/caoh5p7yf5zrznlj54q3p6msiwrhez4iya46ylp4fexsub6q7o2b.py
# Topologically Sorted Source Nodes: [input_618], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_618 => add_503, mul_674, mul_675, sub_224
# Graph fragment:
#   %sub_224 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_224, %unsqueeze_1802), kwargs = {})
#   %mul_674 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_224, %unsqueeze_1805), kwargs = {})
#   %mul_675 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_674, %unsqueeze_1808), kwargs = {})
#   %add_503 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_675, %unsqueeze_1811), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 32)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/mw/cmwokvwatvde2bqaqreqn5mjsnhulcsegajejl2zt4tvjc7hczlq.py
# Topologically Sorted Source Nodes: [input_620, input_621], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_620 => add_505, mul_677, mul_678, sub_225
#   input_621 => clamp_max_164, clamp_min_164
# Graph fragment:
#   %sub_225 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_225, %unsqueeze_1814), kwargs = {})
#   %mul_677 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_225, %unsqueeze_1817), kwargs = {})
#   %mul_678 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_677, %unsqueeze_1820), kwargs = {})
#   %add_505 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_678, %unsqueeze_1823), kwargs = {})
#   %clamp_min_164 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_505, 0.0), kwargs = {})
#   %clamp_max_164 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_164, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 96)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/fl/cfl3o5sp25egx3igoji7k6faikfl3ry5q2q4klpynjb4pwnkwzze.py
# Topologically Sorted Source Nodes: [input_642, cost0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   cost0 => add_522
#   input_642 => add_521, mul_701, mul_702, sub_233
# Graph fragment:
#   %sub_233 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_233, %unsqueeze_1910), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_233, %unsqueeze_1913), kwargs = {})
#   %mul_702 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_701, %unsqueeze_1916), kwargs = {})
#   %add_521 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_702, %unsqueeze_1919), kwargs = {})
#   %add_522 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_521, %add_509), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 32)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/5c/c5c7qqeuozzgx224sgabdsbviyjzpqdleqxtr6lbqrrxknnr3xmg.py
# Topologically Sorted Source Nodes: [input_647, input_648], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_647 => add_526, mul_707, mul_708, sub_235
#   input_648 => clamp_max_171, clamp_min_171
# Graph fragment:
#   %sub_235 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_235, %unsqueeze_1934), kwargs = {})
#   %mul_707 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_235, %unsqueeze_1937), kwargs = {})
#   %mul_708 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_707, %unsqueeze_1940), kwargs = {})
#   %add_526 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_708, %unsqueeze_1943), kwargs = {})
#   %clamp_min_171 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_526, 0.0), kwargs = {})
#   %clamp_max_171 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_171, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/6g/c6gmcj35uhtacmdah5cxga37zivloi4auhykvyxtbsnmf7znlkto.py
# Topologically Sorted Source Nodes: [input_650], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_650 => add_528, mul_710, mul_711, sub_236
# Graph fragment:
#   %sub_236 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_236, %unsqueeze_1946), kwargs = {})
#   %mul_710 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_236, %unsqueeze_1949), kwargs = {})
#   %mul_711 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_710, %unsqueeze_1952), kwargs = {})
#   %add_528 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_711, %unsqueeze_1955), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 64)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxgbiyzqmqr675y6faczubgsyrsa2tx7wpcxik4vniofx4t6v3e3.py
# Topologically Sorted Source Nodes: [input_652, input_653], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_652 => add_530, mul_713, mul_714, sub_237
#   input_653 => clamp_max_172, clamp_min_172
# Graph fragment:
#   %sub_237 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_237, %unsqueeze_1958), kwargs = {})
#   %mul_713 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_237, %unsqueeze_1961), kwargs = {})
#   %mul_714 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_713, %unsqueeze_1964), kwargs = {})
#   %add_530 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_714, %unsqueeze_1967), kwargs = {})
#   %clamp_min_172 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_530, 0.0), kwargs = {})
#   %clamp_max_172 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_172, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 128)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdo26bqsm7o5oqluaokhd6em7ky7uosq3rpcykpfs2u2l6546l3y.py
# Topologically Sorted Source Nodes: [input_663, input_664], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_663 => add_538, mul_725, mul_726, sub_241
#   input_664 => clamp_max_175, clamp_min_175
# Graph fragment:
#   %sub_241 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_241, %unsqueeze_2006), kwargs = {})
#   %mul_725 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_241, %unsqueeze_2009), kwargs = {})
#   %mul_726 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_725, %unsqueeze_2012), kwargs = {})
#   %add_538 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_726, %unsqueeze_2015), kwargs = {})
#   %clamp_min_175 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_538, 0.0), kwargs = {})
#   %clamp_max_175 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_175, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 128)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/2v/c2vbidbdt7323m2boee3opzv3xh27vet7xx7lz7zqdxqebfbjwki.py
# Topologically Sorted Source Nodes: [input_666], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_666 => add_540, mul_728, mul_729, sub_242
# Graph fragment:
#   %sub_242 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_242, %unsqueeze_2018), kwargs = {})
#   %mul_728 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_242, %unsqueeze_2021), kwargs = {})
#   %mul_729 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_728, %unsqueeze_2024), kwargs = {})
#   %add_540 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_729, %unsqueeze_2027), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 128)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/fp/cfpnqcjeak5g3ondpa6hgknsrh635exmu2wxotfil5ed3skeurj2.py
# Topologically Sorted Source Nodes: [input_668, input_669], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_668 => add_542, mul_731, mul_732, sub_243
#   input_669 => clamp_max_176, clamp_min_176
# Graph fragment:
#   %sub_243 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_243, %unsqueeze_2030), kwargs = {})
#   %mul_731 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_243, %unsqueeze_2033), kwargs = {})
#   %mul_732 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_731, %unsqueeze_2036), kwargs = {})
#   %add_542 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_732, %unsqueeze_2039), kwargs = {})
#   %clamp_min_176 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_542, 0.0), kwargs = {})
#   %clamp_max_176 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_176, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 256)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/3h/c3h432bp2vo2nzott7fielhqs22sigq44aonqnouxjegl57fr2wb.py
# Topologically Sorted Source Nodes: [input_676, input_684, add_5, conv5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_5 => add_555
#   conv5 => relu
#   input_676 => add_548, mul_740, mul_741, sub_246
#   input_684 => add_554, mul_749, mul_750, sub_249
# Graph fragment:
#   %sub_246 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_246, %unsqueeze_2066), kwargs = {})
#   %mul_740 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_246, %unsqueeze_2069), kwargs = {})
#   %mul_741 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_740, %unsqueeze_2072), kwargs = {})
#   %add_548 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_741, %unsqueeze_2075), kwargs = {})
#   %sub_249 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_249, %unsqueeze_2102), kwargs = {})
#   %mul_749 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_249, %unsqueeze_2105), kwargs = {})
#   %mul_750 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_749, %unsqueeze_2108), kwargs = {})
#   %add_554 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_750, %unsqueeze_2111), kwargs = {})
#   %add_555 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_548, %add_554), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_555,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x2 = ((xindex // 128) % 64)
    x0 = (xindex % 64)
    x5 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + 64*x5), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/oh/coh5zo2ajjryyuvnswgob4rynjqeu65r2nulle7ufr4fm6dprdcp.py
# Topologically Sorted Source Nodes: [input_686, input_694, add_6, conv6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_6 => add_564
#   conv6 => relu_1
#   input_686 => add_557, mul_752, mul_753, sub_250
#   input_694 => add_563, mul_761, mul_762, sub_253
# Graph fragment:
#   %sub_250 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_250, %unsqueeze_2114), kwargs = {})
#   %mul_752 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_250, %unsqueeze_2117), kwargs = {})
#   %mul_753 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_752, %unsqueeze_2120), kwargs = {})
#   %add_557 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_753, %unsqueeze_2123), kwargs = {})
#   %sub_253 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_253, %unsqueeze_2150), kwargs = {})
#   %mul_761 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_253, %unsqueeze_2153), kwargs = {})
#   %mul_762 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_761, %unsqueeze_2156), kwargs = {})
#   %add_563 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_762, %unsqueeze_2159), kwargs = {})
#   %add_564 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_557, %add_563), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_564,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x2 = ((xindex // 1024) % 32)
    x0 = (xindex % 256)
    x5 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + 256*x5), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/w5/cw53nhsvktjp5dengwq5i3u5ae2hawttxy3fi5ruwfzasurh4isu.py
# Topologically Sorted Source Nodes: [input_696, input_697], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_696 => add_566, mul_764, mul_765, sub_254
#   input_697 => clamp_max_182, clamp_min_182
# Graph fragment:
#   %sub_254 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_254, %unsqueeze_2162), kwargs = {})
#   %mul_764 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_254, %unsqueeze_2165), kwargs = {})
#   %mul_765 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_764, %unsqueeze_2168), kwargs = {})
#   %add_566 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_765, %unsqueeze_2171), kwargs = {})
#   %clamp_min_182 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_566, 0.0), kwargs = {})
#   %clamp_max_182 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_182, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/mz/cmz4neygroefoqyox35osjehwkugfu7iobzecyx5d4xf63hur7ow.py
# Topologically Sorted Source Nodes: [input_699, input_700], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_699 => add_568, mul_767, mul_768, sub_255
#   input_700 => clamp_max_183, clamp_min_183
# Graph fragment:
#   %sub_255 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_255, %unsqueeze_2174), kwargs = {})
#   %mul_767 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_255, %unsqueeze_2177), kwargs = {})
#   %mul_768 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_767, %unsqueeze_2180), kwargs = {})
#   %add_568 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_768, %unsqueeze_2183), kwargs = {})
#   %clamp_min_183 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_568, 0.0), kwargs = {})
#   %clamp_max_183 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_183, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 128) % 64)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ij/cijwyuu4p32xrzsbovzhutzfvvdlfvbzaiwitpb6u274fkzbcr2e.py
# Topologically Sorted Source Nodes: [input_702], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_702 => add_570, mul_770, mul_771, sub_256
# Graph fragment:
#   %sub_256 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_256, %unsqueeze_2186), kwargs = {})
#   %mul_770 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_256, %unsqueeze_2189), kwargs = {})
#   %mul_771 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_770, %unsqueeze_2192), kwargs = {})
#   %add_570 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_771, %unsqueeze_2195), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 128) % 64)
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/hv/chvuku7wl56sdnuuxefsmzg3munkgtu5am76x6zmjyq6m5nq3gp2.py
# Topologically Sorted Source Nodes: [input_704, input_705], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_704 => add_572, mul_773, mul_774, sub_257
#   input_705 => clamp_max_184, clamp_min_184
# Graph fragment:
#   %sub_257 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_257, %unsqueeze_2198), kwargs = {})
#   %mul_773 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_257, %unsqueeze_2201), kwargs = {})
#   %mul_774 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_773, %unsqueeze_2204), kwargs = {})
#   %add_572 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_774, %unsqueeze_2207), kwargs = {})
#   %clamp_min_184 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_572, 0.0), kwargs = {})
#   %clamp_max_184 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_184, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 128) % 128)
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
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ph/cphnofmiff3z73c7yq7x2qwywbthgpvr425cfcgqdr7q3m7zozww.py
# Topologically Sorted Source Nodes: [input_728, input_736, add_7, conv5_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_7 => add_597
#   conv5_1 => relu_2
#   input_728 => add_590, mul_800, mul_801, sub_266
#   input_736 => add_596, mul_809, mul_810, sub_269
# Graph fragment:
#   %sub_266 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_266, %unsqueeze_2306), kwargs = {})
#   %mul_800 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_266, %unsqueeze_2309), kwargs = {})
#   %mul_801 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_800, %unsqueeze_2312), kwargs = {})
#   %add_590 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_801, %unsqueeze_2315), kwargs = {})
#   %sub_269 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_269, %unsqueeze_2342), kwargs = {})
#   %mul_809 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_269, %unsqueeze_2345), kwargs = {})
#   %mul_810 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_809, %unsqueeze_2348), kwargs = {})
#   %add_596 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_810, %unsqueeze_2351), kwargs = {})
#   %add_597 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_590, %add_596), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_597,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 128) % 64)
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


# kernel path: inductor_cache/5c/c5cgiokflso7aiso5u764ceieyfcoqcadyhpofq4c7uojd7cj2w6.py
# Topologically Sorted Source Nodes: [input_738, input_746, add_8, conv6_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_8 => add_606
#   conv6_1 => relu_3
#   input_738 => add_599, mul_812, mul_813, sub_270
#   input_746 => add_605, mul_821, mul_822, sub_273
# Graph fragment:
#   %sub_270 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_270, %unsqueeze_2354), kwargs = {})
#   %mul_812 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_270, %unsqueeze_2357), kwargs = {})
#   %mul_813 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_812, %unsqueeze_2360), kwargs = {})
#   %add_599 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_813, %unsqueeze_2363), kwargs = {})
#   %sub_273 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_273, %unsqueeze_2390), kwargs = {})
#   %mul_821 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_273, %unsqueeze_2393), kwargs = {})
#   %mul_822 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_821, %unsqueeze_2396), kwargs = {})
#   %add_605 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_822, %unsqueeze_2399), kwargs = {})
#   %add_606 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_599, %add_605), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_606,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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


# kernel path: inductor_cache/ne/cneadslclne5cz6olhaxy3vb6orgl2gmnx26h4qjjipg5bgf3e2l.py
# Topologically Sorted Source Nodes: [input_800, input_801], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_800 => add_650, mul_884, mul_885, sub_294
#   input_801 => relu_6
# Graph fragment:
#   %sub_294 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_294, %unsqueeze_2642), kwargs = {})
#   %mul_884 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_294, %unsqueeze_2645), kwargs = {})
#   %mul_885 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_884, %unsqueeze_2648), kwargs = {})
#   %add_650 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_885, %unsqueeze_2651), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_650,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 32)
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


# kernel path: inductor_cache/7j/c7jkvq4ycyhmplhiil5d3ebyq54cpu6fnl66zg4obwy4r267icvg.py
# Topologically Sorted Source Nodes: [cost3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   cost3 => convert_element_type_591
# Graph fragment:
#   %convert_element_type_591 : [num_users=7] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.int64), kwargs = {})
triton_poi_fused__to_copy_42 = async_compile.triton('triton_poi_fused__to_copy_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_42(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wg/cwgplfbebqomlmer5h52fqscvqksvcrgksffdwxtfdv3mg3nc2ly.py
# Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   cost3 => add_652, clamp_max_206
# Graph fragment:
#   %add_652 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_591, 1), kwargs = {})
#   %clamp_max_206 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_652, 3), kwargs = {})
triton_poi_fused_add_clamp_43 = async_compile.triton('triton_poi_fused_add_clamp_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_43(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ku/cku5qzkugxr7niowqoflnoj7jfl47p6r47mbvtte4rs32svpsiok.py
# Topologically Sorted Source Nodes: [cost3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   cost3 => convert_element_type_593
# Graph fragment:
#   %convert_element_type_593 : [num_users=7] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_44 = async_compile.triton('triton_poi_fused__to_copy_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_44(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q4/cq4xi4aeepsytqiubfgrhfhq56giszqa5mtzfmugmfs2y6c4rin4.py
# Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   cost3 => add_654, clamp_max_207
# Graph fragment:
#   %add_654 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_593, 1), kwargs = {})
#   %clamp_max_207 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_654, 15), kwargs = {})
triton_poi_fused_add_clamp_45 = async_compile.triton('triton_poi_fused_add_clamp_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_45(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 15, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lj/clj473drc74qqvvx4qjm7lglrc4lfecpuvxrv4xbad3befqtqkx5.py
# Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   cost3 => add_653, clamp_max_209, clamp_min_207, clamp_min_209, convert_element_type_592, iota_1, mul_887, sub_296, sub_298
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_592 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %add_653 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_592, 0.5), kwargs = {})
#   %mul_887 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_653, 0.25), kwargs = {})
#   %sub_296 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_887, 0.5), kwargs = {})
#   %clamp_min_207 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_296, 0.0), kwargs = {})
#   %sub_298 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_207, %convert_element_type_595), kwargs = {})
#   %clamp_min_209 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_298, 0.0), kwargs = {})
#   %clamp_max_209 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_209, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_46 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_46(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xe/cxeyoctr7mf7isdokpog3ckobw2di2i55it333ivecjo3v6ikncf.py
# Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   cost3 => clamp_max_211, clamp_min_211, sub_306
# Graph fragment:
#   %sub_306 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %convert_element_type_591), kwargs = {})
#   %clamp_min_211 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_306, 0.0), kwargs = {})
#   %clamp_max_211 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_211, 1.0), kwargs = {})
triton_poi_fused_clamp_sub_47 = async_compile.triton('triton_poi_fused_clamp_sub_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_sub_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_sub_47(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = triton_helpers.minimum(tmp12, tmp4)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg64uodgipynpwi6c3rkhnk4szhz2js45ivgwposjdnoxubaoato.py
# Topologically Sorted Source Nodes: [cost3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   cost3 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_657, add_658, add_659, add_660, add_661, add_662, add_663, mul_889, mul_890, mul_891, mul_892, mul_893, mul_894, mul_895, sub_299, sub_300, sub_301, sub_302, sub_304, sub_305, sub_307
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %convert_element_type_591, %convert_element_type_593, %convert_element_type_595]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %convert_element_type_591, %convert_element_type_593, %clamp_max_208]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %convert_element_type_591, %clamp_max_207, %convert_element_type_595]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %convert_element_type_591, %clamp_max_207, %clamp_max_208]), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %clamp_max_206, %convert_element_type_593, %convert_element_type_595]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %clamp_max_206, %convert_element_type_593, %clamp_max_208]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %clamp_max_206, %clamp_max_207, %convert_element_type_595]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_295, [None, None, %clamp_max_206, %clamp_max_207, %clamp_max_208]), kwargs = {})
#   %sub_299 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_889 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_299, %clamp_max_209), kwargs = {})
#   %add_657 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_889), kwargs = {})
#   %sub_300 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_890 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_300, %clamp_max_209), kwargs = {})
#   %add_658 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_890), kwargs = {})
#   %sub_301 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_891 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_301, %clamp_max_209), kwargs = {})
#   %add_659 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_891), kwargs = {})
#   %sub_302 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_892 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_302, %clamp_max_209), kwargs = {})
#   %add_660 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_892), kwargs = {})
#   %sub_304 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_658, %add_657), kwargs = {})
#   %mul_893 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_304, %clamp_max_210), kwargs = {})
#   %add_661 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_657, %mul_893), kwargs = {})
#   %sub_305 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_660, %add_659), kwargs = {})
#   %mul_894 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_305, %clamp_max_210), kwargs = {})
#   %add_662 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_659, %mul_894), kwargs = {})
#   %sub_307 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_662, %add_661), kwargs = {})
#   %mul_895 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_307, %clamp_max_211), kwargs = {})
#   %add_663 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_661, %mul_895), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_48 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 4)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x3 = xindex // 16384
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tl.full([XBLOCK], 16, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp11 = tmp10 + tmp6
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr3 + (tmp13 + 16*tmp9 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp6
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr3 + (tmp18 + 16*tmp9 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp20 = tmp19 - tmp14
    tmp22 = tmp20 * tmp21
    tmp23 = tmp14 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tmp28 = tl.load(in_ptr3 + (tmp13 + 16*tmp9 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (tmp18 + 16*tmp9 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp30 = tmp29 - tmp28
    tmp31 = tmp30 * tmp21
    tmp32 = tmp28 + tmp31
    tmp34 = tmp33 + tmp6
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tmp37 = tl.load(in_ptr3 + (tmp13 + 16*tmp36 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr3 + (tmp18 + 16*tmp36 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp39 = tmp38 - tmp37
    tmp40 = tmp39 * tmp21
    tmp41 = tmp37 + tmp40
    tmp42 = tmp41 - tmp32
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr3 + (tmp13 + 16*tmp36 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr3 + (tmp18 + 16*tmp36 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp47 = tmp46 - tmp45
    tmp48 = tmp47 * tmp21
    tmp49 = tmp45 + tmp48
    tmp50 = tmp49 - tmp23
    tmp51 = tmp50 * tmp43
    tmp52 = tmp32 + tmp44
    tmp53 = tmp23 + tmp51
    tmp54 = tmp53 - tmp52
    tmp56 = tmp54 * tmp55
    tmp57 = tmp52 + tmp56
    tl.store(in_out_ptr0 + (x6), tmp57, None)
''', device_str='cuda')


# kernel path: inductor_cache/5a/c5aksm44wzhubf7s5f63hxa6e4rlcy5a4zgzozhus57uqirir7rn.py
# Topologically Sorted Source Nodes: [pred3], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   pred3 => amax, exp, sub_308
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%squeeze, [1], True), kwargs = {})
#   %sub_308 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_308,), kwargs = {})
triton_poi_fused__softmax_49 = async_compile.triton('triton_poi_fused__softmax_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/yd/cyd4meebxpnaikhn3hmkracna4nknn5k2ohallm4p4zq6s6ikq2h.py
# Topologically Sorted Source Nodes: [pred3], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   pred3 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_50 = async_compile.triton('triton_poi_fused__softmax_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/ru/crujz5arg7a54ekitzhyzd7bluorugvzd44qteb7nkcyflem5tmv.py
# Topologically Sorted Source Nodes: [cost3, disp_values], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   cost3 => iota
#   disp_values => add_664, convert_element_type_596, mul_896
# Graph fragment:
#   %iota : [num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_896 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_664 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_896, 0), kwargs = {})
#   %convert_element_type_596 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_664, torch.float32), kwargs = {})
triton_poi_fused_arange_51 = async_compile.triton('triton_poi_fused_arange_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_51(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/72/c72rujp5zhdxjdqqr42giytcruszpae52xhv734ysnnvfqa4ojof.py
# Topologically Sorted Source Nodes: [mul_1, pred3_1], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_1 => mul_897
#   pred3_1 => sum_2
# Graph fragment:
#   %mul_897 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_4), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_897, [1]), kwargs = {})
triton_poi_fused_mul_sum_52 = async_compile.triton('triton_poi_fused_mul_sum_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_52(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384*x1), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (4096 + x0 + 16384*x1), None)
    tmp5 = tl.load(in_ptr1 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (8192 + x0 + 16384*x1), None)
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (12288 + x0 + 16384*x1), None)
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp7 = tmp4 * tmp6
    tmp8 = tmp3 + tmp7
    tmp12 = tmp9 * tmp11
    tmp13 = tmp8 + tmp12
    tmp17 = tmp14 * tmp16
    tmp18 = tmp13 + tmp17
    tl.store(out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923 = args
    args.clear()
    assert_size_stride(primals_1, (9, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (9, ), (1, ))
    assert_size_stride(primals_4, (9, ), (1, ))
    assert_size_stride(primals_5, (9, ), (1, ))
    assert_size_stride(primals_6, (9, ), (1, ))
    assert_size_stride(primals_7, (9, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_8, (9, ), (1, ))
    assert_size_stride(primals_9, (9, ), (1, ))
    assert_size_stride(primals_10, (9, ), (1, ))
    assert_size_stride(primals_11, (9, ), (1, ))
    assert_size_stride(primals_12, (32, 9, 1, 1), (9, 1, 1, 1))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, ), (1, ))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (96, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_20, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (96, ), (1, ))
    assert_size_stride(primals_24, (96, ), (1, ))
    assert_size_stride(primals_25, (96, ), (1, ))
    assert_size_stride(primals_26, (96, ), (1, ))
    assert_size_stride(primals_27, (32, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_28, (32, ), (1, ))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (32, ), (1, ))
    assert_size_stride(primals_32, (96, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_36, (96, ), (1, ))
    assert_size_stride(primals_37, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_38, (96, ), (1, ))
    assert_size_stride(primals_39, (96, ), (1, ))
    assert_size_stride(primals_40, (96, ), (1, ))
    assert_size_stride(primals_41, (96, ), (1, ))
    assert_size_stride(primals_42, (32, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_43, (32, ), (1, ))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, ), (1, ))
    assert_size_stride(primals_47, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (32, ), (1, ))
    assert_size_stride(primals_50, (32, ), (1, ))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_53, (32, ), (1, ))
    assert_size_stride(primals_54, (32, ), (1, ))
    assert_size_stride(primals_55, (32, ), (1, ))
    assert_size_stride(primals_56, (32, ), (1, ))
    assert_size_stride(primals_57, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_58, (32, ), (1, ))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, ), (1, ))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_63, (32, ), (1, ))
    assert_size_stride(primals_64, (32, ), (1, ))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, ), (1, ))
    assert_size_stride(primals_67, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (32, ), (1, ))
    assert_size_stride(primals_69, (32, ), (1, ))
    assert_size_stride(primals_70, (32, ), (1, ))
    assert_size_stride(primals_71, (32, ), (1, ))
    assert_size_stride(primals_72, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_73, (32, ), (1, ))
    assert_size_stride(primals_74, (32, ), (1, ))
    assert_size_stride(primals_75, (32, ), (1, ))
    assert_size_stride(primals_76, (32, ), (1, ))
    assert_size_stride(primals_77, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_78, (32, ), (1, ))
    assert_size_stride(primals_79, (32, ), (1, ))
    assert_size_stride(primals_80, (32, ), (1, ))
    assert_size_stride(primals_81, (32, ), (1, ))
    assert_size_stride(primals_82, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_83, (32, ), (1, ))
    assert_size_stride(primals_84, (32, ), (1, ))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_86, (32, ), (1, ))
    assert_size_stride(primals_87, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_89, (32, ), (1, ))
    assert_size_stride(primals_90, (32, ), (1, ))
    assert_size_stride(primals_91, (32, ), (1, ))
    assert_size_stride(primals_92, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_93, (32, ), (1, ))
    assert_size_stride(primals_94, (32, ), (1, ))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (32, ), (1, ))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_103, (32, ), (1, ))
    assert_size_stride(primals_104, (32, ), (1, ))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, ), (1, ))
    assert_size_stride(primals_107, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (32, ), (1, ))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (64, ), (1, ))
    assert_size_stride(primals_117, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (64, ), (1, ))
    assert_size_stride(primals_119, (64, ), (1, ))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (64, ), (1, ))
    assert_size_stride(primals_125, (64, ), (1, ))
    assert_size_stride(primals_126, (64, ), (1, ))
    assert_size_stride(primals_127, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_128, (64, ), (1, ))
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (64, ), (1, ))
    assert_size_stride(primals_132, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (64, ), (1, ))
    assert_size_stride(primals_135, (64, ), (1, ))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, ), (1, ))
    assert_size_stride(primals_141, (64, ), (1, ))
    assert_size_stride(primals_142, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (64, ), (1, ))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, ), (1, ))
    assert_size_stride(primals_147, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_158, (64, ), (1, ))
    assert_size_stride(primals_159, (64, ), (1, ))
    assert_size_stride(primals_160, (64, ), (1, ))
    assert_size_stride(primals_161, (64, ), (1, ))
    assert_size_stride(primals_162, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (64, ), (1, ))
    assert_size_stride(primals_165, (64, ), (1, ))
    assert_size_stride(primals_166, (64, ), (1, ))
    assert_size_stride(primals_167, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_168, (64, ), (1, ))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (64, ), (1, ))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (64, ), (1, ))
    assert_size_stride(primals_177, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (64, ), (1, ))
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_183, (64, ), (1, ))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (64, ), (1, ))
    assert_size_stride(primals_192, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (64, ), (1, ))
    assert_size_stride(primals_194, (64, ), (1, ))
    assert_size_stride(primals_195, (64, ), (1, ))
    assert_size_stride(primals_196, (64, ), (1, ))
    assert_size_stride(primals_197, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_198, (64, ), (1, ))
    assert_size_stride(primals_199, (64, ), (1, ))
    assert_size_stride(primals_200, (64, ), (1, ))
    assert_size_stride(primals_201, (64, ), (1, ))
    assert_size_stride(primals_202, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (64, ), (1, ))
    assert_size_stride(primals_205, (64, ), (1, ))
    assert_size_stride(primals_206, (64, ), (1, ))
    assert_size_stride(primals_207, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_208, (64, ), (1, ))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, ), (1, ))
    assert_size_stride(primals_211, (64, ), (1, ))
    assert_size_stride(primals_212, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_213, (64, ), (1, ))
    assert_size_stride(primals_214, (64, ), (1, ))
    assert_size_stride(primals_215, (64, ), (1, ))
    assert_size_stride(primals_216, (64, ), (1, ))
    assert_size_stride(primals_217, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_218, (64, ), (1, ))
    assert_size_stride(primals_219, (64, ), (1, ))
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (64, ), (1, ))
    assert_size_stride(primals_226, (64, ), (1, ))
    assert_size_stride(primals_227, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_228, (64, ), (1, ))
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, ), (1, ))
    assert_size_stride(primals_231, (64, ), (1, ))
    assert_size_stride(primals_232, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_233, (64, ), (1, ))
    assert_size_stride(primals_234, (64, ), (1, ))
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, ), (1, ))
    assert_size_stride(primals_237, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (64, ), (1, ))
    assert_size_stride(primals_241, (64, ), (1, ))
    assert_size_stride(primals_242, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (64, ), (1, ))
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_248, (64, ), (1, ))
    assert_size_stride(primals_249, (64, ), (1, ))
    assert_size_stride(primals_250, (64, ), (1, ))
    assert_size_stride(primals_251, (64, ), (1, ))
    assert_size_stride(primals_252, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (64, ), (1, ))
    assert_size_stride(primals_261, (64, ), (1, ))
    assert_size_stride(primals_262, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_263, (64, ), (1, ))
    assert_size_stride(primals_264, (64, ), (1, ))
    assert_size_stride(primals_265, (64, ), (1, ))
    assert_size_stride(primals_266, (64, ), (1, ))
    assert_size_stride(primals_267, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_268, (64, ), (1, ))
    assert_size_stride(primals_269, (64, ), (1, ))
    assert_size_stride(primals_270, (64, ), (1, ))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (64, ), (1, ))
    assert_size_stride(primals_275, (64, ), (1, ))
    assert_size_stride(primals_276, (64, ), (1, ))
    assert_size_stride(primals_277, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_278, (64, ), (1, ))
    assert_size_stride(primals_279, (64, ), (1, ))
    assert_size_stride(primals_280, (64, ), (1, ))
    assert_size_stride(primals_281, (64, ), (1, ))
    assert_size_stride(primals_282, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_283, (64, ), (1, ))
    assert_size_stride(primals_284, (64, ), (1, ))
    assert_size_stride(primals_285, (64, ), (1, ))
    assert_size_stride(primals_286, (64, ), (1, ))
    assert_size_stride(primals_287, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_288, (64, ), (1, ))
    assert_size_stride(primals_289, (64, ), (1, ))
    assert_size_stride(primals_290, (64, ), (1, ))
    assert_size_stride(primals_291, (64, ), (1, ))
    assert_size_stride(primals_292, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_293, (64, ), (1, ))
    assert_size_stride(primals_294, (64, ), (1, ))
    assert_size_stride(primals_295, (64, ), (1, ))
    assert_size_stride(primals_296, (64, ), (1, ))
    assert_size_stride(primals_297, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_298, (64, ), (1, ))
    assert_size_stride(primals_299, (64, ), (1, ))
    assert_size_stride(primals_300, (64, ), (1, ))
    assert_size_stride(primals_301, (64, ), (1, ))
    assert_size_stride(primals_302, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_303, (64, ), (1, ))
    assert_size_stride(primals_304, (64, ), (1, ))
    assert_size_stride(primals_305, (64, ), (1, ))
    assert_size_stride(primals_306, (64, ), (1, ))
    assert_size_stride(primals_307, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_308, (64, ), (1, ))
    assert_size_stride(primals_309, (64, ), (1, ))
    assert_size_stride(primals_310, (64, ), (1, ))
    assert_size_stride(primals_311, (64, ), (1, ))
    assert_size_stride(primals_312, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (64, ), (1, ))
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (64, ), (1, ))
    assert_size_stride(primals_317, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (64, ), (1, ))
    assert_size_stride(primals_321, (64, ), (1, ))
    assert_size_stride(primals_322, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_323, (64, ), (1, ))
    assert_size_stride(primals_324, (64, ), (1, ))
    assert_size_stride(primals_325, (64, ), (1, ))
    assert_size_stride(primals_326, (64, ), (1, ))
    assert_size_stride(primals_327, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_328, (64, ), (1, ))
    assert_size_stride(primals_329, (64, ), (1, ))
    assert_size_stride(primals_330, (64, ), (1, ))
    assert_size_stride(primals_331, (64, ), (1, ))
    assert_size_stride(primals_332, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_333, (64, ), (1, ))
    assert_size_stride(primals_334, (64, ), (1, ))
    assert_size_stride(primals_335, (64, ), (1, ))
    assert_size_stride(primals_336, (64, ), (1, ))
    assert_size_stride(primals_337, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_338, (64, ), (1, ))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (64, ), (1, ))
    assert_size_stride(primals_345, (64, ), (1, ))
    assert_size_stride(primals_346, (64, ), (1, ))
    assert_size_stride(primals_347, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_348, (64, ), (1, ))
    assert_size_stride(primals_349, (64, ), (1, ))
    assert_size_stride(primals_350, (64, ), (1, ))
    assert_size_stride(primals_351, (64, ), (1, ))
    assert_size_stride(primals_352, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_353, (64, ), (1, ))
    assert_size_stride(primals_354, (64, ), (1, ))
    assert_size_stride(primals_355, (64, ), (1, ))
    assert_size_stride(primals_356, (64, ), (1, ))
    assert_size_stride(primals_357, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_358, (64, ), (1, ))
    assert_size_stride(primals_359, (64, ), (1, ))
    assert_size_stride(primals_360, (64, ), (1, ))
    assert_size_stride(primals_361, (64, ), (1, ))
    assert_size_stride(primals_362, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_363, (64, ), (1, ))
    assert_size_stride(primals_364, (64, ), (1, ))
    assert_size_stride(primals_365, (64, ), (1, ))
    assert_size_stride(primals_366, (64, ), (1, ))
    assert_size_stride(primals_367, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_368, (64, ), (1, ))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (64, ), (1, ))
    assert_size_stride(primals_371, (64, ), (1, ))
    assert_size_stride(primals_372, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_373, (64, ), (1, ))
    assert_size_stride(primals_374, (64, ), (1, ))
    assert_size_stride(primals_375, (64, ), (1, ))
    assert_size_stride(primals_376, (64, ), (1, ))
    assert_size_stride(primals_377, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_378, (64, ), (1, ))
    assert_size_stride(primals_379, (64, ), (1, ))
    assert_size_stride(primals_380, (64, ), (1, ))
    assert_size_stride(primals_381, (64, ), (1, ))
    assert_size_stride(primals_382, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_383, (64, ), (1, ))
    assert_size_stride(primals_384, (64, ), (1, ))
    assert_size_stride(primals_385, (64, ), (1, ))
    assert_size_stride(primals_386, (64, ), (1, ))
    assert_size_stride(primals_387, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_388, (64, ), (1, ))
    assert_size_stride(primals_389, (64, ), (1, ))
    assert_size_stride(primals_390, (64, ), (1, ))
    assert_size_stride(primals_391, (64, ), (1, ))
    assert_size_stride(primals_392, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_393, (64, ), (1, ))
    assert_size_stride(primals_394, (64, ), (1, ))
    assert_size_stride(primals_395, (64, ), (1, ))
    assert_size_stride(primals_396, (64, ), (1, ))
    assert_size_stride(primals_397, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_398, (64, ), (1, ))
    assert_size_stride(primals_399, (64, ), (1, ))
    assert_size_stride(primals_400, (64, ), (1, ))
    assert_size_stride(primals_401, (64, ), (1, ))
    assert_size_stride(primals_402, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_403, (64, ), (1, ))
    assert_size_stride(primals_404, (64, ), (1, ))
    assert_size_stride(primals_405, (64, ), (1, ))
    assert_size_stride(primals_406, (64, ), (1, ))
    assert_size_stride(primals_407, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_408, (64, ), (1, ))
    assert_size_stride(primals_409, (64, ), (1, ))
    assert_size_stride(primals_410, (64, ), (1, ))
    assert_size_stride(primals_411, (64, ), (1, ))
    assert_size_stride(primals_412, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_413, (64, ), (1, ))
    assert_size_stride(primals_414, (64, ), (1, ))
    assert_size_stride(primals_415, (64, ), (1, ))
    assert_size_stride(primals_416, (64, ), (1, ))
    assert_size_stride(primals_417, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_418, (64, ), (1, ))
    assert_size_stride(primals_419, (64, ), (1, ))
    assert_size_stride(primals_420, (64, ), (1, ))
    assert_size_stride(primals_421, (64, ), (1, ))
    assert_size_stride(primals_422, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_423, (64, ), (1, ))
    assert_size_stride(primals_424, (64, ), (1, ))
    assert_size_stride(primals_425, (64, ), (1, ))
    assert_size_stride(primals_426, (64, ), (1, ))
    assert_size_stride(primals_427, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_428, (64, ), (1, ))
    assert_size_stride(primals_429, (64, ), (1, ))
    assert_size_stride(primals_430, (64, ), (1, ))
    assert_size_stride(primals_431, (64, ), (1, ))
    assert_size_stride(primals_432, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_433, (64, ), (1, ))
    assert_size_stride(primals_434, (64, ), (1, ))
    assert_size_stride(primals_435, (64, ), (1, ))
    assert_size_stride(primals_436, (64, ), (1, ))
    assert_size_stride(primals_437, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_438, (128, ), (1, ))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (128, ), (1, ))
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_443, (128, ), (1, ))
    assert_size_stride(primals_444, (128, ), (1, ))
    assert_size_stride(primals_445, (128, ), (1, ))
    assert_size_stride(primals_446, (128, ), (1, ))
    assert_size_stride(primals_447, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_448, (128, ), (1, ))
    assert_size_stride(primals_449, (128, ), (1, ))
    assert_size_stride(primals_450, (128, ), (1, ))
    assert_size_stride(primals_451, (128, ), (1, ))
    assert_size_stride(primals_452, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (128, ), (1, ))
    assert_size_stride(primals_455, (128, ), (1, ))
    assert_size_stride(primals_456, (128, ), (1, ))
    assert_size_stride(primals_457, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_458, (128, ), (1, ))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (128, ), (1, ))
    assert_size_stride(primals_461, (128, ), (1, ))
    assert_size_stride(primals_462, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_463, (128, ), (1, ))
    assert_size_stride(primals_464, (128, ), (1, ))
    assert_size_stride(primals_465, (128, ), (1, ))
    assert_size_stride(primals_466, (128, ), (1, ))
    assert_size_stride(primals_467, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_468, (128, ), (1, ))
    assert_size_stride(primals_469, (128, ), (1, ))
    assert_size_stride(primals_470, (128, ), (1, ))
    assert_size_stride(primals_471, (128, ), (1, ))
    assert_size_stride(primals_472, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_473, (128, ), (1, ))
    assert_size_stride(primals_474, (128, ), (1, ))
    assert_size_stride(primals_475, (128, ), (1, ))
    assert_size_stride(primals_476, (128, ), (1, ))
    assert_size_stride(primals_477, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_478, (128, ), (1, ))
    assert_size_stride(primals_479, (128, ), (1, ))
    assert_size_stride(primals_480, (128, ), (1, ))
    assert_size_stride(primals_481, (128, ), (1, ))
    assert_size_stride(primals_482, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_483, (128, ), (1, ))
    assert_size_stride(primals_484, (128, ), (1, ))
    assert_size_stride(primals_485, (128, ), (1, ))
    assert_size_stride(primals_486, (128, ), (1, ))
    assert_size_stride(primals_487, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_488, (128, ), (1, ))
    assert_size_stride(primals_489, (128, ), (1, ))
    assert_size_stride(primals_490, (128, ), (1, ))
    assert_size_stride(primals_491, (128, ), (1, ))
    assert_size_stride(primals_492, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_493, (128, ), (1, ))
    assert_size_stride(primals_494, (128, ), (1, ))
    assert_size_stride(primals_495, (128, ), (1, ))
    assert_size_stride(primals_496, (128, ), (1, ))
    assert_size_stride(primals_497, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_498, (128, ), (1, ))
    assert_size_stride(primals_499, (128, ), (1, ))
    assert_size_stride(primals_500, (128, ), (1, ))
    assert_size_stride(primals_501, (128, ), (1, ))
    assert_size_stride(primals_502, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_503, (128, ), (1, ))
    assert_size_stride(primals_504, (128, ), (1, ))
    assert_size_stride(primals_505, (128, ), (1, ))
    assert_size_stride(primals_506, (128, ), (1, ))
    assert_size_stride(primals_507, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_508, (128, ), (1, ))
    assert_size_stride(primals_509, (128, ), (1, ))
    assert_size_stride(primals_510, (128, ), (1, ))
    assert_size_stride(primals_511, (128, ), (1, ))
    assert_size_stride(primals_512, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_513, (128, ), (1, ))
    assert_size_stride(primals_514, (128, ), (1, ))
    assert_size_stride(primals_515, (128, ), (1, ))
    assert_size_stride(primals_516, (128, ), (1, ))
    assert_size_stride(primals_517, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_518, (128, ), (1, ))
    assert_size_stride(primals_519, (128, ), (1, ))
    assert_size_stride(primals_520, (128, ), (1, ))
    assert_size_stride(primals_521, (128, ), (1, ))
    assert_size_stride(primals_522, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_523, (128, ), (1, ))
    assert_size_stride(primals_524, (128, ), (1, ))
    assert_size_stride(primals_525, (128, ), (1, ))
    assert_size_stride(primals_526, (128, ), (1, ))
    assert_size_stride(primals_527, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_528, (128, ), (1, ))
    assert_size_stride(primals_529, (128, ), (1, ))
    assert_size_stride(primals_530, (128, ), (1, ))
    assert_size_stride(primals_531, (128, ), (1, ))
    assert_size_stride(primals_532, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_533, (128, ), (1, ))
    assert_size_stride(primals_534, (128, ), (1, ))
    assert_size_stride(primals_535, (128, ), (1, ))
    assert_size_stride(primals_536, (128, ), (1, ))
    assert_size_stride(primals_537, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_538, (128, ), (1, ))
    assert_size_stride(primals_539, (128, ), (1, ))
    assert_size_stride(primals_540, (128, ), (1, ))
    assert_size_stride(primals_541, (128, ), (1, ))
    assert_size_stride(primals_542, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_543, (128, ), (1, ))
    assert_size_stride(primals_544, (128, ), (1, ))
    assert_size_stride(primals_545, (128, ), (1, ))
    assert_size_stride(primals_546, (128, ), (1, ))
    assert_size_stride(primals_547, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_548, (128, ), (1, ))
    assert_size_stride(primals_549, (128, ), (1, ))
    assert_size_stride(primals_550, (128, ), (1, ))
    assert_size_stride(primals_551, (128, ), (1, ))
    assert_size_stride(primals_552, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_553, (128, ), (1, ))
    assert_size_stride(primals_554, (128, ), (1, ))
    assert_size_stride(primals_555, (128, ), (1, ))
    assert_size_stride(primals_556, (128, ), (1, ))
    assert_size_stride(primals_557, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_558, (120, 40, 1, 1, 1), (40, 1, 1, 1, 1))
    assert_size_stride(primals_559, (120, ), (1, ))
    assert_size_stride(primals_560, (120, ), (1, ))
    assert_size_stride(primals_561, (120, ), (1, ))
    assert_size_stride(primals_562, (120, ), (1, ))
    assert_size_stride(primals_563, (120, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_564, (120, ), (1, ))
    assert_size_stride(primals_565, (120, ), (1, ))
    assert_size_stride(primals_566, (120, ), (1, ))
    assert_size_stride(primals_567, (120, ), (1, ))
    assert_size_stride(primals_568, (32, 120, 1, 1, 1), (120, 1, 1, 1, 1))
    assert_size_stride(primals_569, (32, ), (1, ))
    assert_size_stride(primals_570, (32, ), (1, ))
    assert_size_stride(primals_571, (32, ), (1, ))
    assert_size_stride(primals_572, (32, ), (1, ))
    assert_size_stride(primals_573, (96, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_574, (96, ), (1, ))
    assert_size_stride(primals_575, (96, ), (1, ))
    assert_size_stride(primals_576, (96, ), (1, ))
    assert_size_stride(primals_577, (96, ), (1, ))
    assert_size_stride(primals_578, (96, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_579, (96, ), (1, ))
    assert_size_stride(primals_580, (96, ), (1, ))
    assert_size_stride(primals_581, (96, ), (1, ))
    assert_size_stride(primals_582, (96, ), (1, ))
    assert_size_stride(primals_583, (32, 96, 1, 1, 1), (96, 1, 1, 1, 1))
    assert_size_stride(primals_584, (32, ), (1, ))
    assert_size_stride(primals_585, (32, ), (1, ))
    assert_size_stride(primals_586, (32, ), (1, ))
    assert_size_stride(primals_587, (32, ), (1, ))
    assert_size_stride(primals_588, (96, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_589, (96, ), (1, ))
    assert_size_stride(primals_590, (96, ), (1, ))
    assert_size_stride(primals_591, (96, ), (1, ))
    assert_size_stride(primals_592, (96, ), (1, ))
    assert_size_stride(primals_593, (96, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_594, (96, ), (1, ))
    assert_size_stride(primals_595, (96, ), (1, ))
    assert_size_stride(primals_596, (96, ), (1, ))
    assert_size_stride(primals_597, (96, ), (1, ))
    assert_size_stride(primals_598, (32, 96, 1, 1, 1), (96, 1, 1, 1, 1))
    assert_size_stride(primals_599, (32, ), (1, ))
    assert_size_stride(primals_600, (32, ), (1, ))
    assert_size_stride(primals_601, (32, ), (1, ))
    assert_size_stride(primals_602, (32, ), (1, ))
    assert_size_stride(primals_603, (96, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_604, (96, ), (1, ))
    assert_size_stride(primals_605, (96, ), (1, ))
    assert_size_stride(primals_606, (96, ), (1, ))
    assert_size_stride(primals_607, (96, ), (1, ))
    assert_size_stride(primals_608, (96, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_609, (96, ), (1, ))
    assert_size_stride(primals_610, (96, ), (1, ))
    assert_size_stride(primals_611, (96, ), (1, ))
    assert_size_stride(primals_612, (96, ), (1, ))
    assert_size_stride(primals_613, (32, 96, 1, 1, 1), (96, 1, 1, 1, 1))
    assert_size_stride(primals_614, (32, ), (1, ))
    assert_size_stride(primals_615, (32, ), (1, ))
    assert_size_stride(primals_616, (32, ), (1, ))
    assert_size_stride(primals_617, (32, ), (1, ))
    assert_size_stride(primals_618, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_619, (64, ), (1, ))
    assert_size_stride(primals_620, (64, ), (1, ))
    assert_size_stride(primals_621, (64, ), (1, ))
    assert_size_stride(primals_622, (64, ), (1, ))
    assert_size_stride(primals_623, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_624, (64, ), (1, ))
    assert_size_stride(primals_625, (64, ), (1, ))
    assert_size_stride(primals_626, (64, ), (1, ))
    assert_size_stride(primals_627, (64, ), (1, ))
    assert_size_stride(primals_628, (64, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_629, (64, ), (1, ))
    assert_size_stride(primals_630, (64, ), (1, ))
    assert_size_stride(primals_631, (64, ), (1, ))
    assert_size_stride(primals_632, (64, ), (1, ))
    assert_size_stride(primals_633, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_634, (128, ), (1, ))
    assert_size_stride(primals_635, (128, ), (1, ))
    assert_size_stride(primals_636, (128, ), (1, ))
    assert_size_stride(primals_637, (128, ), (1, ))
    assert_size_stride(primals_638, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_639, (128, ), (1, ))
    assert_size_stride(primals_640, (128, ), (1, ))
    assert_size_stride(primals_641, (128, ), (1, ))
    assert_size_stride(primals_642, (128, ), (1, ))
    assert_size_stride(primals_643, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_644, (64, ), (1, ))
    assert_size_stride(primals_645, (64, ), (1, ))
    assert_size_stride(primals_646, (64, ), (1, ))
    assert_size_stride(primals_647, (64, ), (1, ))
    assert_size_stride(primals_648, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_649, (128, ), (1, ))
    assert_size_stride(primals_650, (128, ), (1, ))
    assert_size_stride(primals_651, (128, ), (1, ))
    assert_size_stride(primals_652, (128, ), (1, ))
    assert_size_stride(primals_653, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_654, (128, ), (1, ))
    assert_size_stride(primals_655, (128, ), (1, ))
    assert_size_stride(primals_656, (128, ), (1, ))
    assert_size_stride(primals_657, (128, ), (1, ))
    assert_size_stride(primals_658, (128, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_659, (128, ), (1, ))
    assert_size_stride(primals_660, (128, ), (1, ))
    assert_size_stride(primals_661, (128, ), (1, ))
    assert_size_stride(primals_662, (128, ), (1, ))
    assert_size_stride(primals_663, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_664, (256, ), (1, ))
    assert_size_stride(primals_665, (256, ), (1, ))
    assert_size_stride(primals_666, (256, ), (1, ))
    assert_size_stride(primals_667, (256, ), (1, ))
    assert_size_stride(primals_668, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_669, (256, ), (1, ))
    assert_size_stride(primals_670, (256, ), (1, ))
    assert_size_stride(primals_671, (256, ), (1, ))
    assert_size_stride(primals_672, (256, ), (1, ))
    assert_size_stride(primals_673, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(primals_674, (128, ), (1, ))
    assert_size_stride(primals_675, (128, ), (1, ))
    assert_size_stride(primals_676, (128, ), (1, ))
    assert_size_stride(primals_677, (128, ), (1, ))
    assert_size_stride(primals_678, (128, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_679, (64, ), (1, ))
    assert_size_stride(primals_680, (64, ), (1, ))
    assert_size_stride(primals_681, (64, ), (1, ))
    assert_size_stride(primals_682, (64, ), (1, ))
    assert_size_stride(primals_683, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_684, (128, ), (1, ))
    assert_size_stride(primals_685, (128, ), (1, ))
    assert_size_stride(primals_686, (128, ), (1, ))
    assert_size_stride(primals_687, (128, ), (1, ))
    assert_size_stride(primals_688, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_689, (128, ), (1, ))
    assert_size_stride(primals_690, (128, ), (1, ))
    assert_size_stride(primals_691, (128, ), (1, ))
    assert_size_stride(primals_692, (128, ), (1, ))
    assert_size_stride(primals_693, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_694, (64, ), (1, ))
    assert_size_stride(primals_695, (64, ), (1, ))
    assert_size_stride(primals_696, (64, ), (1, ))
    assert_size_stride(primals_697, (64, ), (1, ))
    assert_size_stride(primals_698, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_699, (32, ), (1, ))
    assert_size_stride(primals_700, (32, ), (1, ))
    assert_size_stride(primals_701, (32, ), (1, ))
    assert_size_stride(primals_702, (32, ), (1, ))
    assert_size_stride(primals_703, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_704, (64, ), (1, ))
    assert_size_stride(primals_705, (64, ), (1, ))
    assert_size_stride(primals_706, (64, ), (1, ))
    assert_size_stride(primals_707, (64, ), (1, ))
    assert_size_stride(primals_708, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_709, (64, ), (1, ))
    assert_size_stride(primals_710, (64, ), (1, ))
    assert_size_stride(primals_711, (64, ), (1, ))
    assert_size_stride(primals_712, (64, ), (1, ))
    assert_size_stride(primals_713, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_714, (32, ), (1, ))
    assert_size_stride(primals_715, (32, ), (1, ))
    assert_size_stride(primals_716, (32, ), (1, ))
    assert_size_stride(primals_717, (32, ), (1, ))
    assert_size_stride(primals_718, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_719, (64, ), (1, ))
    assert_size_stride(primals_720, (64, ), (1, ))
    assert_size_stride(primals_721, (64, ), (1, ))
    assert_size_stride(primals_722, (64, ), (1, ))
    assert_size_stride(primals_723, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_724, (64, ), (1, ))
    assert_size_stride(primals_725, (64, ), (1, ))
    assert_size_stride(primals_726, (64, ), (1, ))
    assert_size_stride(primals_727, (64, ), (1, ))
    assert_size_stride(primals_728, (64, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_729, (64, ), (1, ))
    assert_size_stride(primals_730, (64, ), (1, ))
    assert_size_stride(primals_731, (64, ), (1, ))
    assert_size_stride(primals_732, (64, ), (1, ))
    assert_size_stride(primals_733, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_734, (128, ), (1, ))
    assert_size_stride(primals_735, (128, ), (1, ))
    assert_size_stride(primals_736, (128, ), (1, ))
    assert_size_stride(primals_737, (128, ), (1, ))
    assert_size_stride(primals_738, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_739, (128, ), (1, ))
    assert_size_stride(primals_740, (128, ), (1, ))
    assert_size_stride(primals_741, (128, ), (1, ))
    assert_size_stride(primals_742, (128, ), (1, ))
    assert_size_stride(primals_743, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_744, (64, ), (1, ))
    assert_size_stride(primals_745, (64, ), (1, ))
    assert_size_stride(primals_746, (64, ), (1, ))
    assert_size_stride(primals_747, (64, ), (1, ))
    assert_size_stride(primals_748, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_749, (128, ), (1, ))
    assert_size_stride(primals_750, (128, ), (1, ))
    assert_size_stride(primals_751, (128, ), (1, ))
    assert_size_stride(primals_752, (128, ), (1, ))
    assert_size_stride(primals_753, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_754, (128, ), (1, ))
    assert_size_stride(primals_755, (128, ), (1, ))
    assert_size_stride(primals_756, (128, ), (1, ))
    assert_size_stride(primals_757, (128, ), (1, ))
    assert_size_stride(primals_758, (128, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_759, (128, ), (1, ))
    assert_size_stride(primals_760, (128, ), (1, ))
    assert_size_stride(primals_761, (128, ), (1, ))
    assert_size_stride(primals_762, (128, ), (1, ))
    assert_size_stride(primals_763, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_764, (256, ), (1, ))
    assert_size_stride(primals_765, (256, ), (1, ))
    assert_size_stride(primals_766, (256, ), (1, ))
    assert_size_stride(primals_767, (256, ), (1, ))
    assert_size_stride(primals_768, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_769, (256, ), (1, ))
    assert_size_stride(primals_770, (256, ), (1, ))
    assert_size_stride(primals_771, (256, ), (1, ))
    assert_size_stride(primals_772, (256, ), (1, ))
    assert_size_stride(primals_773, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(primals_774, (128, ), (1, ))
    assert_size_stride(primals_775, (128, ), (1, ))
    assert_size_stride(primals_776, (128, ), (1, ))
    assert_size_stride(primals_777, (128, ), (1, ))
    assert_size_stride(primals_778, (128, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_779, (64, ), (1, ))
    assert_size_stride(primals_780, (64, ), (1, ))
    assert_size_stride(primals_781, (64, ), (1, ))
    assert_size_stride(primals_782, (64, ), (1, ))
    assert_size_stride(primals_783, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_784, (128, ), (1, ))
    assert_size_stride(primals_785, (128, ), (1, ))
    assert_size_stride(primals_786, (128, ), (1, ))
    assert_size_stride(primals_787, (128, ), (1, ))
    assert_size_stride(primals_788, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_789, (128, ), (1, ))
    assert_size_stride(primals_790, (128, ), (1, ))
    assert_size_stride(primals_791, (128, ), (1, ))
    assert_size_stride(primals_792, (128, ), (1, ))
    assert_size_stride(primals_793, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_794, (64, ), (1, ))
    assert_size_stride(primals_795, (64, ), (1, ))
    assert_size_stride(primals_796, (64, ), (1, ))
    assert_size_stride(primals_797, (64, ), (1, ))
    assert_size_stride(primals_798, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_799, (32, ), (1, ))
    assert_size_stride(primals_800, (32, ), (1, ))
    assert_size_stride(primals_801, (32, ), (1, ))
    assert_size_stride(primals_802, (32, ), (1, ))
    assert_size_stride(primals_803, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_804, (64, ), (1, ))
    assert_size_stride(primals_805, (64, ), (1, ))
    assert_size_stride(primals_806, (64, ), (1, ))
    assert_size_stride(primals_807, (64, ), (1, ))
    assert_size_stride(primals_808, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_809, (64, ), (1, ))
    assert_size_stride(primals_810, (64, ), (1, ))
    assert_size_stride(primals_811, (64, ), (1, ))
    assert_size_stride(primals_812, (64, ), (1, ))
    assert_size_stride(primals_813, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_814, (32, ), (1, ))
    assert_size_stride(primals_815, (32, ), (1, ))
    assert_size_stride(primals_816, (32, ), (1, ))
    assert_size_stride(primals_817, (32, ), (1, ))
    assert_size_stride(primals_818, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_819, (64, ), (1, ))
    assert_size_stride(primals_820, (64, ), (1, ))
    assert_size_stride(primals_821, (64, ), (1, ))
    assert_size_stride(primals_822, (64, ), (1, ))
    assert_size_stride(primals_823, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_824, (64, ), (1, ))
    assert_size_stride(primals_825, (64, ), (1, ))
    assert_size_stride(primals_826, (64, ), (1, ))
    assert_size_stride(primals_827, (64, ), (1, ))
    assert_size_stride(primals_828, (64, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_829, (64, ), (1, ))
    assert_size_stride(primals_830, (64, ), (1, ))
    assert_size_stride(primals_831, (64, ), (1, ))
    assert_size_stride(primals_832, (64, ), (1, ))
    assert_size_stride(primals_833, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_834, (128, ), (1, ))
    assert_size_stride(primals_835, (128, ), (1, ))
    assert_size_stride(primals_836, (128, ), (1, ))
    assert_size_stride(primals_837, (128, ), (1, ))
    assert_size_stride(primals_838, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_839, (128, ), (1, ))
    assert_size_stride(primals_840, (128, ), (1, ))
    assert_size_stride(primals_841, (128, ), (1, ))
    assert_size_stride(primals_842, (128, ), (1, ))
    assert_size_stride(primals_843, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_844, (64, ), (1, ))
    assert_size_stride(primals_845, (64, ), (1, ))
    assert_size_stride(primals_846, (64, ), (1, ))
    assert_size_stride(primals_847, (64, ), (1, ))
    assert_size_stride(primals_848, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_849, (128, ), (1, ))
    assert_size_stride(primals_850, (128, ), (1, ))
    assert_size_stride(primals_851, (128, ), (1, ))
    assert_size_stride(primals_852, (128, ), (1, ))
    assert_size_stride(primals_853, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_854, (128, ), (1, ))
    assert_size_stride(primals_855, (128, ), (1, ))
    assert_size_stride(primals_856, (128, ), (1, ))
    assert_size_stride(primals_857, (128, ), (1, ))
    assert_size_stride(primals_858, (128, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_859, (128, ), (1, ))
    assert_size_stride(primals_860, (128, ), (1, ))
    assert_size_stride(primals_861, (128, ), (1, ))
    assert_size_stride(primals_862, (128, ), (1, ))
    assert_size_stride(primals_863, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_864, (256, ), (1, ))
    assert_size_stride(primals_865, (256, ), (1, ))
    assert_size_stride(primals_866, (256, ), (1, ))
    assert_size_stride(primals_867, (256, ), (1, ))
    assert_size_stride(primals_868, (256, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_869, (256, ), (1, ))
    assert_size_stride(primals_870, (256, ), (1, ))
    assert_size_stride(primals_871, (256, ), (1, ))
    assert_size_stride(primals_872, (256, ), (1, ))
    assert_size_stride(primals_873, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(primals_874, (128, ), (1, ))
    assert_size_stride(primals_875, (128, ), (1, ))
    assert_size_stride(primals_876, (128, ), (1, ))
    assert_size_stride(primals_877, (128, ), (1, ))
    assert_size_stride(primals_878, (128, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_879, (64, ), (1, ))
    assert_size_stride(primals_880, (64, ), (1, ))
    assert_size_stride(primals_881, (64, ), (1, ))
    assert_size_stride(primals_882, (64, ), (1, ))
    assert_size_stride(primals_883, (128, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_884, (128, ), (1, ))
    assert_size_stride(primals_885, (128, ), (1, ))
    assert_size_stride(primals_886, (128, ), (1, ))
    assert_size_stride(primals_887, (128, ), (1, ))
    assert_size_stride(primals_888, (128, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_889, (128, ), (1, ))
    assert_size_stride(primals_890, (128, ), (1, ))
    assert_size_stride(primals_891, (128, ), (1, ))
    assert_size_stride(primals_892, (128, ), (1, ))
    assert_size_stride(primals_893, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_894, (64, ), (1, ))
    assert_size_stride(primals_895, (64, ), (1, ))
    assert_size_stride(primals_896, (64, ), (1, ))
    assert_size_stride(primals_897, (64, ), (1, ))
    assert_size_stride(primals_898, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_899, (32, ), (1, ))
    assert_size_stride(primals_900, (32, ), (1, ))
    assert_size_stride(primals_901, (32, ), (1, ))
    assert_size_stride(primals_902, (32, ), (1, ))
    assert_size_stride(primals_903, (64, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_904, (64, ), (1, ))
    assert_size_stride(primals_905, (64, ), (1, ))
    assert_size_stride(primals_906, (64, ), (1, ))
    assert_size_stride(primals_907, (64, ), (1, ))
    assert_size_stride(primals_908, (64, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_909, (64, ), (1, ))
    assert_size_stride(primals_910, (64, ), (1, ))
    assert_size_stride(primals_911, (64, ), (1, ))
    assert_size_stride(primals_912, (64, ), (1, ))
    assert_size_stride(primals_913, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_914, (32, ), (1, ))
    assert_size_stride(primals_915, (32, ), (1, ))
    assert_size_stride(primals_916, (32, ), (1, ))
    assert_size_stride(primals_917, (32, ), (1, ))
    assert_size_stride(primals_918, (32, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_919, (32, ), (1, ))
    assert_size_stride(primals_920, (32, ), (1, ))
    assert_size_stride(primals_921, (32, ), (1, ))
    assert_size_stride(primals_922, (32, ), (1, ))
    assert_size_stride(primals_923, (1, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 9, 64, 64), (36864, 4096, 64, 1))
        buf1 = empty_strided_cuda((4, 9, 64, 64), (36864, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=9, bias=None)
        assert_size_stride(buf2, (4, 9, 32, 32), (9216, 1024, 32, 1))
        buf3 = empty_strided_cuda((4, 9, 32, 32), (9216, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1.run(buf2, primals_8, primals_9, primals_10, primals_11, buf3, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf5 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf4, primals_13, primals_14, primals_15, primals_16, buf5, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf7 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf6, primals_18, primals_19, primals_20, primals_21, buf7, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf8, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf9 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf8, primals_23, primals_24, primals_25, primals_26, buf9, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf11 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf5, buf10, primals_28, primals_29, primals_30, primals_31, buf11, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf13 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf12, primals_33, primals_34, primals_35, primals_36, buf13, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf14, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf15 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf14, primals_38, primals_39, primals_40, primals_41, buf15, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf17 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf11, buf16, primals_43, primals_44, primals_45, primals_46, buf17, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf18, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf19 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf18, primals_48, primals_49, primals_50, primals_51, buf19, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf21 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf20, primals_53, primals_54, primals_55, primals_56, buf21, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf22, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf23 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf22, primals_58, primals_59, primals_60, primals_61, buf23, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf25 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_6.run(buf24, primals_63, primals_64, primals_65, primals_66, buf17, buf25, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf26, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf27 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf26, primals_68, primals_69, primals_70, primals_71, buf27, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf29 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf28, primals_73, primals_74, primals_75, primals_76, buf29, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf30, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf31 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf30, primals_78, primals_79, primals_80, primals_81, buf31, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf33 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_48, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_6.run(buf32, primals_83, primals_84, primals_85, primals_86, buf25, buf33, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf34, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf35 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf34, primals_88, primals_89, primals_90, primals_91, buf35, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf37 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_53, input_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf36, primals_93, primals_94, primals_95, primals_96, buf37, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf38, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf39 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf38, primals_98, primals_99, primals_100, primals_101, buf39, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf41 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_6.run(buf40, primals_103, primals_104, primals_105, primals_106, buf33, buf41, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_107, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf42, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf43 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_61, input_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7.run(buf42, primals_108, primals_109, primals_110, primals_111, buf43, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf45 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf44, primals_113, primals_114, primals_115, primals_116, buf45, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf46, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf47 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_67, input_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf46, primals_118, primals_119, primals_120, primals_121, buf47, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf41, primals_127, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf50 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_70, input_72, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf48, primals_123, primals_124, primals_125, primals_126, buf49, primals_128, primals_129, primals_130, primals_131, buf50, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf51, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf52 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_74, input_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf51, primals_133, primals_134, primals_135, primals_136, buf52, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf54 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf53, primals_138, primals_139, primals_140, primals_141, buf54, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf55, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf56 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf55, primals_143, primals_144, primals_145, primals_146, buf56, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf58 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_83, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf57, primals_148, primals_149, primals_150, primals_151, buf50, buf58, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf59, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf60 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_85, input_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf59, primals_153, primals_154, primals_155, primals_156, buf60, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf62 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_88, input_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf61, primals_158, primals_159, primals_160, primals_161, buf62, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf63, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf64 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_91, input_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf63, primals_163, primals_164, primals_165, primals_166, buf64, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf66 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_94, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf65, primals_168, primals_169, primals_170, primals_171, buf58, buf66, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf67, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf68 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_96, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf67, primals_173, primals_174, primals_175, primals_176, buf68, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf70 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_99, input_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf69, primals_178, primals_179, primals_180, primals_181, buf70, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf71, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf72 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_102, input_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf71, primals_183, primals_184, primals_185, primals_186, buf72, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf74 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_105, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf73, primals_188, primals_189, primals_190, primals_191, buf66, buf74, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf75, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf76 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_107, input_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf75, primals_193, primals_194, primals_195, primals_196, buf76, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf78 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf77, primals_198, primals_199, primals_200, primals_201, buf78, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf79, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf80 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_113, input_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf79, primals_203, primals_204, primals_205, primals_206, buf80, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf82 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_116, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf81, primals_208, primals_209, primals_210, primals_211, buf74, buf82, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_117], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf83, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf84 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_118, input_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf83, primals_213, primals_214, primals_215, primals_216, buf84, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf86 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_121, input_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf85, primals_218, primals_219, primals_220, primals_221, buf86, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_123], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf87, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf88 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_124, input_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf87, primals_223, primals_224, primals_225, primals_226, buf88, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf90 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_127, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf89, primals_228, primals_229, primals_230, primals_231, buf82, buf90, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf91, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf92 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_129, input_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf91, primals_233, primals_234, primals_235, primals_236, buf92, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf94 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_132, input_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf93, primals_238, primals_239, primals_240, primals_241, buf94, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf95, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf96 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_135, input_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf95, primals_243, primals_244, primals_245, primals_246, buf96, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_137], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf98 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_138, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf97, primals_248, primals_249, primals_250, primals_251, buf90, buf98, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf99, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf100 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_140, input_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf99, primals_253, primals_254, primals_255, primals_256, buf100, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf102 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_143, input_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf101, primals_258, primals_259, primals_260, primals_261, buf102, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf103, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf104 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_146, input_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf103, primals_263, primals_264, primals_265, primals_266, buf104, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf106 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_149, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf105, primals_268, primals_269, primals_270, primals_271, buf98, buf106, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf107, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf108 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_151, input_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf107, primals_273, primals_274, primals_275, primals_276, buf108, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf110 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_154, input_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf109, primals_278, primals_279, primals_280, primals_281, buf110, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf111, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf112 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_157, input_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf111, primals_283, primals_284, primals_285, primals_286, buf112, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf114 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_160, out_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf113, primals_288, primals_289, primals_290, primals_291, buf106, buf114, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf115, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf116 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_162, input_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf115, primals_293, primals_294, primals_295, primals_296, buf116, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf118 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_165, input_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf117, primals_298, primals_299, primals_300, primals_301, buf118, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_167], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf119, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf120 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_168, input_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf119, primals_303, primals_304, primals_305, primals_306, buf120, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_170], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf122 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_171, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf121, primals_308, primals_309, primals_310, primals_311, buf114, buf122, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf123, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf124 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_173, input_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf123, primals_313, primals_314, primals_315, primals_316, buf124, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_175], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf126 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_176, input_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf125, primals_318, primals_319, primals_320, primals_321, buf126, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf127, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf128 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_179, input_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf127, primals_323, primals_324, primals_325, primals_326, buf128, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf130 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_182, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf129, primals_328, primals_329, primals_330, primals_331, buf122, buf130, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_183], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf131, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf132 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_184, input_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf131, primals_333, primals_334, primals_335, primals_336, buf132, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_186], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf134 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_187, input_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf133, primals_338, primals_339, primals_340, primals_341, buf134, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf135, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf136 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_190, input_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf135, primals_343, primals_344, primals_345, primals_346, buf136, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_192], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf138 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_193, out_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf137, primals_348, primals_349, primals_350, primals_351, buf130, buf138, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf139, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf140 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_195, input_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf139, primals_353, primals_354, primals_355, primals_356, buf140, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_197], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf142 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_198, input_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf141, primals_358, primals_359, primals_360, primals_361, buf142, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_200], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf143, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf144 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_201, input_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf143, primals_363, primals_364, primals_365, primals_366, buf144, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_203], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf146 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_204, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf145, primals_368, primals_369, primals_370, primals_371, buf138, buf146, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_205], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf147, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf148 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_206, input_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf147, primals_373, primals_374, primals_375, primals_376, buf148, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_208], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_377, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf150 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_209, input_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf149, primals_378, primals_379, primals_380, primals_381, buf150, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_211], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf151, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf152 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_212, input_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf151, primals_383, primals_384, primals_385, primals_386, buf152, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf154 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_215, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf153, primals_388, primals_389, primals_390, primals_391, buf146, buf154, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_216], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf155, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf156 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_217, input_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf155, primals_393, primals_394, primals_395, primals_396, buf156, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_219], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf158 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_220, input_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf157, primals_398, primals_399, primals_400, primals_401, buf158, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_222], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf159, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf160 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_223, input_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf159, primals_403, primals_404, primals_405, primals_406, buf160, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_225], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_407, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf162 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_226, out_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf161, primals_408, primals_409, primals_410, primals_411, buf154, buf162, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_227], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf163, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf164 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_228, input_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf163, primals_413, primals_414, primals_415, primals_416, buf164, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_230], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_417, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf166 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_231, input_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf165, primals_418, primals_419, primals_420, primals_421, buf166, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_233], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf167, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf168 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_234, input_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf167, primals_423, primals_424, primals_425, primals_426, buf168, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_236], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_427, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf170 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_237, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf169, primals_428, primals_429, primals_430, primals_431, buf162, buf170, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf171, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf172 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_239, input_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf171, primals_433, primals_434, primals_435, primals_436, buf172, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_241], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf174 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_242, input_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf173, primals_438, primals_439, primals_440, primals_441, buf174, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_244], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf175, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf176 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_245, input_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf175, primals_443, primals_444, primals_445, primals_446, buf176, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_247], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 128, 16, 16), (32768, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf170, primals_452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf179 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_248, input_250, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf177, primals_448, primals_449, primals_450, primals_451, buf178, primals_453, primals_454, primals_455, primals_456, buf179, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_457, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf180, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf181 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_252, input_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf180, primals_458, primals_459, primals_460, primals_461, buf181, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_254], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf183 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_255, input_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf182, primals_463, primals_464, primals_465, primals_466, buf183, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_257], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_467, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf184, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf185 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_258, input_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf184, primals_468, primals_469, primals_470, primals_471, buf185, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_260], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf187 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_261, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf186, primals_473, primals_474, primals_475, primals_476, buf179, buf187, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_262], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_477, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf188, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf189 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_263, input_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf188, primals_478, primals_479, primals_480, primals_481, buf189, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_265], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_482, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf191 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_266, input_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf190, primals_483, primals_484, primals_485, primals_486, buf191, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_268], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_487, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf192, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf193 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_269, input_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf192, primals_488, primals_489, primals_490, primals_491, buf193, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_271], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf195 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_272, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf194, primals_493, primals_494, primals_495, primals_496, buf187, buf195, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_497, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf196, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf197 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_274, input_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf196, primals_498, primals_499, primals_500, primals_501, buf197, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_276], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf199 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_277, input_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf198, primals_503, primals_504, primals_505, primals_506, buf199, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_279], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_507, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf200, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf201 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_280, input_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf200, primals_508, primals_509, primals_510, primals_511, buf201, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_282], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf203 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_283, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf202, primals_513, primals_514, primals_515, primals_516, buf195, buf203, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_284], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_517, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf204, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf205 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_285, input_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf204, primals_518, primals_519, primals_520, primals_521, buf205, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf207 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_288, input_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf206, primals_523, primals_524, primals_525, primals_526, buf207, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_290], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_527, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf208, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf209 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_291, input_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf208, primals_528, primals_529, primals_530, primals_531, buf209, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_293], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_532, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf211 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_294, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf210, primals_533, primals_534, primals_535, primals_536, buf203, buf211, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_295], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_537, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf212, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf213 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_296, input_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf212, primals_538, primals_539, primals_540, primals_541, buf213, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_298], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_542, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf215 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_299, input_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf214, primals_543, primals_544, primals_545, primals_546, buf215, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_301], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_547, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf216, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf217 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_302, input_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf216, primals_548, primals_549, primals_550, primals_551, buf217, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_304], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_552, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf219 = empty_strided_cuda((4, 320, 16, 16), (81920, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [feature_volume], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf170, buf195, buf218, primals_553, primals_554, primals_555, primals_556, buf211, buf219, 327680, grid=grid(327680), stream=stream0)
        # Topologically Sorted Source Nodes: [input_306], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(primals_557, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 9, 64, 64), (36864, 4096, 64, 1))
        buf222 = empty_strided_cuda((4, 9, 64, 64), (36864, 4096, 64, 1), torch.float32)
        buf768 = empty_strided_cuda((4, 9, 64, 64), (36864, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_307, input_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_15.run(buf220, primals_3, primals_4, primals_5, primals_6, buf222, buf768, 147456, grid=grid(147456), stream=stream0)
        # Topologically Sorted Source Nodes: [input_309], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=9, bias=None)
        assert_size_stride(buf223, (4, 9, 32, 32), (9216, 1024, 32, 1))
        buf225 = empty_strided_cuda((4, 9, 32, 32), (9216, 1024, 32, 1), torch.float32)
        buf767 = empty_strided_cuda((4, 9, 32, 32), (9216, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_310, input_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_16.run(buf223, primals_8, primals_9, primals_10, primals_11, buf225, buf767, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [input_312], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf227 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_313], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf226, primals_13, primals_14, primals_15, primals_16, buf227, 131072, grid=grid(131072), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf230 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        buf766 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_315, input_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17.run(buf228, primals_18, primals_19, primals_20, primals_21, buf230, buf766, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_317], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf231, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf233 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        buf765 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_318, input_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17.run(buf231, primals_23, primals_24, primals_25, primals_26, buf233, buf765, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf235 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_321, input_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf227, buf234, primals_28, primals_29, primals_30, primals_31, buf235, 131072, grid=grid(131072), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [input_323], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf238 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        buf764 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_324, input_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17.run(buf236, primals_33, primals_34, primals_35, primals_36, buf238, buf764, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_326], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf239, (4, 96, 32, 32), (98304, 1024, 32, 1))
        buf241 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.float32)
        buf763 = empty_strided_cuda((4, 96, 32, 32), (98304, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_327, input_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_17.run(buf239, primals_38, primals_39, primals_40, primals_41, buf241, buf763, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_329], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf243 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_330, input_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf235, buf242, primals_43, primals_44, primals_45, primals_46, buf243, 131072, grid=grid(131072), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [input_332], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf244, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf246 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf762 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_333, input_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf244, primals_48, primals_49, primals_50, primals_51, buf246, buf762, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_335], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf249 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf761 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_336, input_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf247, primals_53, primals_54, primals_55, primals_56, buf249, buf761, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_338], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf250, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf252 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf760 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_339, input_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf250, primals_58, primals_59, primals_60, primals_61, buf252, buf760, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_341], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf254 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_342, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_6.run(buf253, primals_63, primals_64, primals_65, primals_66, buf243, buf254, 131072, grid=grid(131072), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [input_343], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf255, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf257 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf759 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_344, input_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf255, primals_68, primals_69, primals_70, primals_71, buf257, buf759, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_346], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf260 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf758 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_347, input_348], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf258, primals_73, primals_74, primals_75, primals_76, buf260, buf758, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_349], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf261, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf263 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf757 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_350, input_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf261, primals_78, primals_79, primals_80, primals_81, buf263, buf757, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_352], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf265 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_353, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_6.run(buf264, primals_83, primals_84, primals_85, primals_86, buf254, buf265, 131072, grid=grid(131072), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [input_354], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf266, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf268 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf756 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_355, input_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf266, primals_88, primals_89, primals_90, primals_91, buf268, buf756, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_357], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf271 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf755 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_358, input_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf269, primals_93, primals_94, primals_95, primals_96, buf271, buf755, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_360], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf272, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf274 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf754 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_361, input_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_18.run(buf272, primals_98, primals_99, primals_100, primals_101, buf274, buf754, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_363], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf276 = empty_strided_cuda((4, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_364, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_6.run(buf275, primals_103, primals_104, primals_105, primals_106, buf265, buf276, 131072, grid=grid(131072), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [input_365], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_107, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf277, (4, 32, 16, 16), (8192, 256, 16, 1))
        buf279 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf753 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_366, input_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_19.run(buf277, primals_108, primals_109, primals_110, primals_111, buf279, buf753, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_368], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf282 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf752 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_369, input_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf280, primals_113, primals_114, primals_115, primals_116, buf282, buf752, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_371], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf283, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf285 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf751 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_372, input_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf283, primals_118, primals_119, primals_120, primals_121, buf285, buf751, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_374], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 64, 16, 16), (16384, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_376], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf276, primals_127, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf288 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_375, input_377, out_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf286, primals_123, primals_124, primals_125, primals_126, buf287, primals_128, primals_129, primals_130, primals_131, buf288, 65536, grid=grid(65536), stream=stream0)
        del primals_126
        del primals_131
        # Topologically Sorted Source Nodes: [input_378], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf289, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf291 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf750 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_379, input_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf289, primals_133, primals_134, primals_135, primals_136, buf291, buf750, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_381], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf294 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf749 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_382, input_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf292, primals_138, primals_139, primals_140, primals_141, buf294, buf749, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_384], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf295, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf297 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf748 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_385, input_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf295, primals_143, primals_144, primals_145, primals_146, buf297, buf748, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_387], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf299 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_388, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf298, primals_148, primals_149, primals_150, primals_151, buf288, buf299, 65536, grid=grid(65536), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [input_389], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf300, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf302 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf747 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_390, input_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf300, primals_153, primals_154, primals_155, primals_156, buf302, buf747, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_392], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf305 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf746 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_393, input_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf303, primals_158, primals_159, primals_160, primals_161, buf305, buf746, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_395], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf306, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf308 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf745 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_396, input_397], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf306, primals_163, primals_164, primals_165, primals_166, buf308, buf745, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_398], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf310 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_399, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf309, primals_168, primals_169, primals_170, primals_171, buf299, buf310, 65536, grid=grid(65536), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [input_400], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf311, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf313 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf744 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_401, input_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf311, primals_173, primals_174, primals_175, primals_176, buf313, buf744, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_403], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf316 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf743 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_404, input_405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf314, primals_178, primals_179, primals_180, primals_181, buf316, buf743, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_406], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf317, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf319 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf742 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_407, input_408], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf317, primals_183, primals_184, primals_185, primals_186, buf319, buf742, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_409], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf321 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_410, out_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf320, primals_188, primals_189, primals_190, primals_191, buf310, buf321, 65536, grid=grid(65536), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [input_411], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf322, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf324 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf741 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_412, input_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf322, primals_193, primals_194, primals_195, primals_196, buf324, buf741, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_414], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf327 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf740 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_415, input_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf325, primals_198, primals_199, primals_200, primals_201, buf327, buf740, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_417], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf328, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf330 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf739 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_418, input_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf328, primals_203, primals_204, primals_205, primals_206, buf330, buf739, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_420], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf332 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_421, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf331, primals_208, primals_209, primals_210, primals_211, buf321, buf332, 65536, grid=grid(65536), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [input_422], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf333, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf335 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf738 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_423, input_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf333, primals_213, primals_214, primals_215, primals_216, buf335, buf738, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_425], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf338 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf737 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_426, input_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf336, primals_218, primals_219, primals_220, primals_221, buf338, buf737, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_428], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf339, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf341 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf736 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_429, input_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf339, primals_223, primals_224, primals_225, primals_226, buf341, buf736, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_431], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf343 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_432, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf342, primals_228, primals_229, primals_230, primals_231, buf332, buf343, 65536, grid=grid(65536), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [input_433], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf344, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf346 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf735 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_434, input_435], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf344, primals_233, primals_234, primals_235, primals_236, buf346, buf735, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_436], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf349 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf734 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_437, input_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf347, primals_238, primals_239, primals_240, primals_241, buf349, buf734, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_439], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf350, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf352 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf733 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_440, input_441], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf350, primals_243, primals_244, primals_245, primals_246, buf352, buf733, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_442], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf354 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_443, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf353, primals_248, primals_249, primals_250, primals_251, buf343, buf354, 65536, grid=grid(65536), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [input_444], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf355, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf357 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf732 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_445, input_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf355, primals_253, primals_254, primals_255, primals_256, buf357, buf732, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_447], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf360 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf731 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_448, input_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf358, primals_258, primals_259, primals_260, primals_261, buf360, buf731, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_450], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf361, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf363 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf730 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_451, input_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf361, primals_263, primals_264, primals_265, primals_266, buf363, buf730, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_453], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf365 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_454, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf364, primals_268, primals_269, primals_270, primals_271, buf354, buf365, 65536, grid=grid(65536), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [input_455], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf366, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf368 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf729 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_456, input_457], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf366, primals_273, primals_274, primals_275, primals_276, buf368, buf729, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_458], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf371 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf728 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_459, input_460], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf369, primals_278, primals_279, primals_280, primals_281, buf371, buf728, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_461], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf372, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf374 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf727 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_462, input_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf372, primals_283, primals_284, primals_285, primals_286, buf374, buf727, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_464], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf376 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_465, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf375, primals_288, primals_289, primals_290, primals_291, buf365, buf376, 65536, grid=grid(65536), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [input_466], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf377, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf379 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf726 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_467, input_468], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf377, primals_293, primals_294, primals_295, primals_296, buf379, buf726, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_469], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf382 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf725 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_470, input_471], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf380, primals_298, primals_299, primals_300, primals_301, buf382, buf725, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_472], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf383, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf385 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf724 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_473, input_474], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf383, primals_303, primals_304, primals_305, primals_306, buf385, buf724, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_475], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf387 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_476, out_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf386, primals_308, primals_309, primals_310, primals_311, buf376, buf387, 65536, grid=grid(65536), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [input_477], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf387, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf388, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf390 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf723 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_478, input_479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf388, primals_313, primals_314, primals_315, primals_316, buf390, buf723, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_480], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf393 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf722 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_481, input_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf391, primals_318, primals_319, primals_320, primals_321, buf393, buf722, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_483], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf394, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf396 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf721 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_484, input_485], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf394, primals_323, primals_324, primals_325, primals_326, buf396, buf721, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_486], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf398 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_487, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf397, primals_328, primals_329, primals_330, primals_331, buf387, buf398, 65536, grid=grid(65536), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [input_488], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf399, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf401 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf720 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_489, input_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf399, primals_333, primals_334, primals_335, primals_336, buf401, buf720, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_491], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf404 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf719 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_492, input_493], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf402, primals_338, primals_339, primals_340, primals_341, buf404, buf719, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_494], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf404, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf405, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf407 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf718 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_495, input_496], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf405, primals_343, primals_344, primals_345, primals_346, buf407, buf718, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_497], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf409 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_498, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf408, primals_348, primals_349, primals_350, primals_351, buf398, buf409, 65536, grid=grid(65536), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [input_499], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf410, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf412 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf717 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_500, input_501], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf410, primals_353, primals_354, primals_355, primals_356, buf412, buf717, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_502], Original ATen: [aten.convolution]
        buf413 = extern_kernels.convolution(buf412, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf413, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf415 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf716 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_503, input_504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf413, primals_358, primals_359, primals_360, primals_361, buf415, buf716, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_505], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf416, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf418 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf715 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_506, input_507], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf416, primals_363, primals_364, primals_365, primals_366, buf418, buf715, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_508], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf420 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_509, out_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf419, primals_368, primals_369, primals_370, primals_371, buf409, buf420, 65536, grid=grid(65536), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [input_510], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf421, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf423 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf714 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_511, input_512], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf421, primals_373, primals_374, primals_375, primals_376, buf423, buf714, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_513], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_377, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf426 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf713 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_514, input_515], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf424, primals_378, primals_379, primals_380, primals_381, buf426, buf713, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_516], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf427, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf429 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf712 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_517, input_518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf427, primals_383, primals_384, primals_385, primals_386, buf429, buf712, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_519], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf431 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_520, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf430, primals_388, primals_389, primals_390, primals_391, buf420, buf431, 65536, grid=grid(65536), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [input_521], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf432, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf434 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf711 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_522, input_523], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf432, primals_393, primals_394, primals_395, primals_396, buf434, buf711, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_524], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, primals_397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf437 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf710 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_525, input_526], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf435, primals_398, primals_399, primals_400, primals_401, buf437, buf710, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_527], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf438, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf440 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf709 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_528, input_529], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf438, primals_403, primals_404, primals_405, primals_406, buf440, buf709, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_530], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_407, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf442 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_531, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf441, primals_408, primals_409, primals_410, primals_411, buf431, buf442, 65536, grid=grid(65536), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [input_532], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf443, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf445 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf708 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_533, input_534], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf443, primals_413, primals_414, primals_415, primals_416, buf445, buf708, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_535], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_417, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf448 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf707 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_536, input_537], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf446, primals_418, primals_419, primals_420, primals_421, buf448, buf707, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_538], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf449, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf451 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf706 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_539, input_540], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf449, primals_423, primals_424, primals_425, primals_426, buf451, buf706, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_541], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, primals_427, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf453 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_542, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf452, primals_428, primals_429, primals_430, primals_431, buf442, buf453, 65536, grid=grid(65536), stream=stream0)
        del primals_431
        # Topologically Sorted Source Nodes: [input_543], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf454, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf456 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf705 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_544, input_545], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_20.run(buf454, primals_433, primals_434, primals_435, primals_436, buf456, buf705, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_546], Original ATen: [aten.convolution]
        buf457 = extern_kernels.convolution(buf456, primals_437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf457, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf459 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf704 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_547, input_548], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf457, primals_438, primals_439, primals_440, primals_441, buf459, buf704, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_549], Original ATen: [aten.convolution]
        buf460 = extern_kernels.convolution(buf459, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf460, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf462 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf703 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_550, input_551], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf460, primals_443, primals_444, primals_445, primals_446, buf462, buf703, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_552], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 128, 16, 16), (32768, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_554], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf453, primals_452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf465 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_553, input_555, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf463, primals_448, primals_449, primals_450, primals_451, buf464, primals_453, primals_454, primals_455, primals_456, buf465, 131072, grid=grid(131072), stream=stream0)
        del primals_451
        del primals_456
        # Topologically Sorted Source Nodes: [input_556], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, primals_457, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf466, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf468 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf702 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_557, input_558], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf466, primals_458, primals_459, primals_460, primals_461, buf468, buf702, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_559], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf468, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf471 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf701 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_560, input_561], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf469, primals_463, primals_464, primals_465, primals_466, buf471, buf701, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_562], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, primals_467, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf472, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf474 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf700 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_563, input_564], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf472, primals_468, primals_469, primals_470, primals_471, buf474, buf700, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_565], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf476 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_566, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf475, primals_473, primals_474, primals_475, primals_476, buf465, buf476, 131072, grid=grid(131072), stream=stream0)
        del primals_476
        # Topologically Sorted Source Nodes: [input_567], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_477, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf477, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf479 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf699 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_568, input_569], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf477, primals_478, primals_479, primals_480, primals_481, buf479, buf699, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_570], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, primals_482, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf482 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf698 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_571, input_572], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf480, primals_483, primals_484, primals_485, primals_486, buf482, buf698, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_573], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_487, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf483, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf485 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf697 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_574, input_575], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf483, primals_488, primals_489, primals_490, primals_491, buf485, buf697, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_576], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf487 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_577, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf486, primals_493, primals_494, primals_495, primals_496, buf476, buf487, 131072, grid=grid(131072), stream=stream0)
        del primals_496
        # Topologically Sorted Source Nodes: [input_578], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf487, primals_497, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf488, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf490 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf696 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_579, input_580], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf488, primals_498, primals_499, primals_500, primals_501, buf490, buf696, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_581], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, primals_502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf493 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf695 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_582, input_583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf491, primals_503, primals_504, primals_505, primals_506, buf493, buf695, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_584], Original ATen: [aten.convolution]
        buf494 = extern_kernels.convolution(buf493, primals_507, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf494, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf496 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf694 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_585, input_586], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf494, primals_508, primals_509, primals_510, primals_511, buf496, buf694, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_587], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf498 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_588, out_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf497, primals_513, primals_514, primals_515, primals_516, buf487, buf498, 131072, grid=grid(131072), stream=stream0)
        del primals_516
        # Topologically Sorted Source Nodes: [input_589], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, primals_517, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf499, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf501 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf693 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_590, input_591], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf499, primals_518, primals_519, primals_520, primals_521, buf501, buf693, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_592], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf504 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf692 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_593, input_594], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf502, primals_523, primals_524, primals_525, primals_526, buf504, buf692, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_595], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, primals_527, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf505, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf507 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf691 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_596, input_597], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf505, primals_528, primals_529, primals_530, primals_531, buf507, buf691, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_598], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, primals_532, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf509 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_599, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf508, primals_533, primals_534, primals_535, primals_536, buf498, buf509, 131072, grid=grid(131072), stream=stream0)
        del primals_536
        # Topologically Sorted Source Nodes: [input_600], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, primals_537, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf510, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf512 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf690 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_601, input_602], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf510, primals_538, primals_539, primals_540, primals_541, buf512, buf690, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_603], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_542, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf515 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf689 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_604, input_605], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf513, primals_543, primals_544, primals_545, primals_546, buf515, buf689, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_606], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, primals_547, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf516, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf518 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf688 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_607, input_608], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_hardtanh_backward_21.run(buf516, primals_548, primals_549, primals_550, primals_551, buf518, buf688, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_609], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf518, primals_552, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf520 = empty_strided_cuda((4, 320, 16, 16), (81920, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [feature_volume_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf453, buf487, buf519, primals_553, primals_554, primals_555, primals_556, buf509, buf520, 327680, grid=grid(327680), stream=stream0)
        del primals_556
        buf521 = empty_strided_cuda((4, 40, 16, 16), (10240, 256, 16, 1), torch.float32)
        buf522 = reinterpret_tensor(buf521, (4, 40, 1, 16, 16), (10240, 256, 256, 16, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [volume, cost], Original ATen: [aten.new_zeros, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_new_zeros_22.run(buf522, buf219, buf520, 40960, 8, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [input_611], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_558, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (4, 120, 1, 16, 16), (30720, 256, 256, 16, 1))
        buf524 = empty_strided_cuda((4, 120, 1, 16, 16), (30720, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_612, input_613], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_23.run(buf523, primals_559, primals_560, primals_561, primals_562, buf524, 122880, grid=grid(122880), stream=stream0)
        # Topologically Sorted Source Nodes: [input_614], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf524, primals_563, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=120, bias=None)
        assert_size_stride(buf525, (4, 120, 1, 16, 16), (30720, 256, 256, 16, 1))
        buf526 = empty_strided_cuda((4, 120, 1, 16, 16), (30720, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_615, input_616], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_23.run(buf525, primals_564, primals_565, primals_566, primals_567, buf526, 122880, grid=grid(122880), stream=stream0)
        # Topologically Sorted Source Nodes: [input_617], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, primals_568, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (4, 32, 1, 16, 16), (8192, 256, 256, 16, 1))
        buf528 = empty_strided_cuda((4, 32, 1, 16, 16), (8192, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_618], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf527, primals_569, primals_570, primals_571, primals_572, buf528, 32768, grid=grid(32768), stream=stream0)
        del primals_572
        # Topologically Sorted Source Nodes: [input_619], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf528, primals_573, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (4, 96, 1, 16, 16), (24576, 256, 256, 16, 1))
        buf530 = empty_strided_cuda((4, 96, 1, 16, 16), (24576, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_620, input_621], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25.run(buf529, primals_574, primals_575, primals_576, primals_577, buf530, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_622], Original ATen: [aten.convolution]
        buf531 = extern_kernels.convolution(buf530, primals_578, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=96, bias=None)
        assert_size_stride(buf531, (4, 96, 1, 16, 16), (24576, 256, 256, 16, 1))
        buf532 = empty_strided_cuda((4, 96, 1, 16, 16), (24576, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_623, input_624], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25.run(buf531, primals_579, primals_580, primals_581, primals_582, buf532, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_625], Original ATen: [aten.convolution]
        buf533 = extern_kernels.convolution(buf532, primals_583, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (4, 32, 1, 16, 16), (8192, 256, 256, 16, 1))
        buf534 = empty_strided_cuda((4, 32, 1, 16, 16), (8192, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_626], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf533, primals_584, primals_585, primals_586, primals_587, buf534, 32768, grid=grid(32768), stream=stream0)
        del primals_587
        # Topologically Sorted Source Nodes: [input_627], Original ATen: [aten.convolution]
        buf535 = extern_kernels.convolution(buf534, primals_588, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf535, (4, 96, 1, 16, 16), (24576, 256, 256, 16, 1))
        buf536 = empty_strided_cuda((4, 96, 1, 16, 16), (24576, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_628, input_629], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25.run(buf535, primals_589, primals_590, primals_591, primals_592, buf536, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_630], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf536, primals_593, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=96, bias=None)
        assert_size_stride(buf537, (4, 96, 1, 16, 16), (24576, 256, 256, 16, 1))
        buf538 = empty_strided_cuda((4, 96, 1, 16, 16), (24576, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_631, input_632], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25.run(buf537, primals_594, primals_595, primals_596, primals_597, buf538, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_633], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_598, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (4, 32, 1, 16, 16), (8192, 256, 256, 16, 1))
        buf540 = empty_strided_cuda((4, 32, 1, 16, 16), (8192, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_634], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf539, primals_599, primals_600, primals_601, primals_602, buf540, 32768, grid=grid(32768), stream=stream0)
        del primals_602
        # Topologically Sorted Source Nodes: [input_635], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, primals_603, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (4, 96, 1, 16, 16), (24576, 256, 256, 16, 1))
        buf542 = empty_strided_cuda((4, 96, 1, 16, 16), (24576, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_636, input_637], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25.run(buf541, primals_604, primals_605, primals_606, primals_607, buf542, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_638], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, primals_608, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=96, bias=None)
        assert_size_stride(buf543, (4, 96, 1, 16, 16), (24576, 256, 256, 16, 1))
        buf544 = empty_strided_cuda((4, 96, 1, 16, 16), (24576, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_639, input_640], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_25.run(buf543, primals_609, primals_610, primals_611, primals_612, buf544, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_641], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf544, primals_613, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (4, 32, 1, 16, 16), (8192, 256, 256, 16, 1))
        buf546 = empty_strided_cuda((4, 32, 1, 16, 16), (8192, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_642, cost0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf545, primals_614, primals_615, primals_616, primals_617, buf534, buf546, 32768, grid=grid(32768), stream=stream0)
        del primals_617
        # Topologically Sorted Source Nodes: [input_643], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, primals_618, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (4, 64, 1, 16, 16), (16384, 256, 256, 16, 1))
        buf548 = empty_strided_cuda((4, 64, 1, 16, 16), (16384, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_644, input_645], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf547, primals_619, primals_620, primals_621, primals_622, buf548, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_646], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(buf548, primals_623, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf549, (4, 64, 1, 8, 8), (4096, 64, 64, 8, 1))
        buf550 = empty_strided_cuda((4, 64, 1, 8, 8), (4096, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_647, input_648], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_27.run(buf549, primals_624, primals_625, primals_626, primals_627, buf550, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_649], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_628, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 64, 1, 8, 8), (4096, 64, 64, 8, 1))
        buf552 = empty_strided_cuda((4, 64, 1, 8, 8), (4096, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_650], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf551, primals_629, primals_630, primals_631, primals_632, buf552, 16384, grid=grid(16384), stream=stream0)
        del primals_632
        # Topologically Sorted Source Nodes: [input_651], Original ATen: [aten.convolution]
        buf553 = extern_kernels.convolution(buf552, primals_633, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf553, (4, 128, 1, 8, 8), (8192, 64, 64, 8, 1))
        buf554 = empty_strided_cuda((4, 128, 1, 8, 8), (8192, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_652, input_653], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29.run(buf553, primals_634, primals_635, primals_636, primals_637, buf554, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_654], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(buf554, primals_638, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf555, (4, 128, 1, 8, 8), (8192, 64, 64, 8, 1))
        buf556 = empty_strided_cuda((4, 128, 1, 8, 8), (8192, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_655, input_656], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29.run(buf555, primals_639, primals_640, primals_641, primals_642, buf556, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_657], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, primals_643, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (4, 64, 1, 8, 8), (4096, 64, 64, 8, 1))
        buf558 = empty_strided_cuda((4, 64, 1, 8, 8), (4096, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_658], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf557, primals_644, primals_645, primals_646, primals_647, buf558, 16384, grid=grid(16384), stream=stream0)
        del primals_647
        # Topologically Sorted Source Nodes: [input_659], Original ATen: [aten.convolution]
        buf559 = extern_kernels.convolution(buf558, primals_648, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf559, (4, 128, 1, 8, 8), (8192, 64, 64, 8, 1))
        buf560 = empty_strided_cuda((4, 128, 1, 8, 8), (8192, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_660, input_661], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29.run(buf559, primals_649, primals_650, primals_651, primals_652, buf560, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_662], Original ATen: [aten.convolution]
        buf561 = extern_kernels.convolution(buf560, primals_653, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf561, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf562 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_663, input_664], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_30.run(buf561, primals_654, primals_655, primals_656, primals_657, buf562, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_665], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, primals_658, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf564 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_666], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf563, primals_659, primals_660, primals_661, primals_662, buf564, 8192, grid=grid(8192), stream=stream0)
        del primals_662
        # Topologically Sorted Source Nodes: [input_667], Original ATen: [aten.convolution]
        buf565 = extern_kernels.convolution(buf564, primals_663, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf565, (4, 256, 1, 4, 4), (4096, 16, 16, 4, 1))
        buf566 = empty_strided_cuda((4, 256, 1, 4, 4), (4096, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_668, input_669], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32.run(buf565, primals_664, primals_665, primals_666, primals_667, buf566, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_670], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, primals_668, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf567, (4, 256, 1, 4, 4), (4096, 16, 16, 4, 1))
        buf568 = empty_strided_cuda((4, 256, 1, 4, 4), (4096, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_671, input_672], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32.run(buf567, primals_669, primals_670, primals_671, primals_672, buf568, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_673], Original ATen: [aten.convolution]
        buf569 = extern_kernels.convolution(buf568, primals_673, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf569, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf570 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_674], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf569, primals_674, primals_675, primals_676, primals_677, buf570, 8192, grid=grid(8192), stream=stream0)
        del primals_677
        # Topologically Sorted Source Nodes: [input_675], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, primals_678, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf571, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_677], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf558, primals_683, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (4, 128, 1, 8, 8), (8192, 64, 64, 8, 1))
        buf573 = empty_strided_cuda((4, 128, 1, 8, 8), (8192, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_678, input_679], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29.run(buf572, primals_684, primals_685, primals_686, primals_687, buf573, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_680], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_688, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf574, (4, 128, 1, 8, 8), (8192, 64, 64, 8, 1))
        buf575 = empty_strided_cuda((4, 128, 1, 8, 8), (8192, 64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_681, input_682], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_29.run(buf574, primals_689, primals_690, primals_691, primals_692, buf575, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_683], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_693, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 64, 1, 8, 8), (4096, 64, 64, 8, 1))
        buf577 = empty_strided_cuda((4, 64, 1, 8, 8), (4096, 64, 16384, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_684], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf576, primals_694, primals_695, primals_696, primals_697, buf577, 16384, grid=grid(16384), stream=stream0)
        del primals_697
        buf578 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_676, input_684, add_5, conv5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf571, primals_679, primals_680, primals_681, primals_682, buf577, buf578, 32768, grid=grid(32768), stream=stream0)
        del primals_682
        # Topologically Sorted Source Nodes: [input_685], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, primals_698, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf579, (4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_687], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf546, primals_703, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (4, 64, 1, 16, 16), (16384, 256, 256, 16, 1))
        buf581 = empty_strided_cuda((4, 64, 1, 16, 16), (16384, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_688, input_689], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf580, primals_704, primals_705, primals_706, primals_707, buf581, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_690], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, primals_708, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf582, (4, 64, 1, 16, 16), (16384, 256, 256, 16, 1))
        buf583 = empty_strided_cuda((4, 64, 1, 16, 16), (16384, 256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_691, input_692], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf582, primals_709, primals_710, primals_711, primals_712, buf583, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_693], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf583, primals_713, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (4, 32, 1, 16, 16), (8192, 256, 256, 16, 1))
        buf585 = empty_strided_cuda((4, 32, 1, 16, 16), (8192, 256, 32768, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_694], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf584, primals_714, primals_715, primals_716, primals_717, buf585, 32768, grid=grid(32768), stream=stream0)
        del primals_717
        buf586 = empty_strided_cuda((4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_686, input_694, add_6, conv6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_34.run(buf579, primals_699, primals_700, primals_701, primals_702, buf585, buf586, 131072, grid=grid(131072), stream=stream0)
        del primals_702
        # Topologically Sorted Source Nodes: [input_695], Original ATen: [aten.convolution]
        buf587 = extern_kernels.convolution(buf586, primals_718, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf587, (4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1))
        buf588 = empty_strided_cuda((4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_696, input_697], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35.run(buf587, primals_719, primals_720, primals_721, primals_722, buf588, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_698], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, primals_723, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf589, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf590 = reinterpret_tensor(buf585, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), 0); del buf585  # reuse
        # Topologically Sorted Source Nodes: [input_699, input_700], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_36.run(buf589, primals_724, primals_725, primals_726, primals_727, buf590, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_701], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf590, primals_728, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf592 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_702], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf591, primals_729, primals_730, primals_731, primals_732, buf592, 32768, grid=grid(32768), stream=stream0)
        del primals_732
        # Topologically Sorted Source Nodes: [input_703], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, primals_733, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf594 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_704, input_705], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf593, primals_734, primals_735, primals_736, primals_737, buf594, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_706], Original ATen: [aten.convolution]
        buf595 = extern_kernels.convolution(buf594, primals_738, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf595, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf596 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_707, input_708], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf595, primals_739, primals_740, primals_741, primals_742, buf596, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_709], Original ATen: [aten.convolution]
        buf597 = extern_kernels.convolution(buf596, primals_743, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf597, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf598 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_710], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf597, primals_744, primals_745, primals_746, primals_747, buf598, 32768, grid=grid(32768), stream=stream0)
        del primals_747
        # Topologically Sorted Source Nodes: [input_711], Original ATen: [aten.convolution]
        buf599 = extern_kernels.convolution(buf598, primals_748, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf599, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf600 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_712, input_713], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf599, primals_749, primals_750, primals_751, primals_752, buf600, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_714], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, primals_753, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf601, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf602 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_715, input_716], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_30.run(buf601, primals_754, primals_755, primals_756, primals_757, buf602, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_717], Original ATen: [aten.convolution]
        buf603 = extern_kernels.convolution(buf602, primals_758, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf603, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf604 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_718], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf603, primals_759, primals_760, primals_761, primals_762, buf604, 8192, grid=grid(8192), stream=stream0)
        del primals_762
        # Topologically Sorted Source Nodes: [input_719], Original ATen: [aten.convolution]
        buf605 = extern_kernels.convolution(buf604, primals_763, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf605, (4, 256, 1, 4, 4), (4096, 16, 16, 4, 1))
        buf606 = reinterpret_tensor(buf577, (4, 256, 1, 4, 4), (4096, 16, 16, 4, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [input_720, input_721], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32.run(buf605, primals_764, primals_765, primals_766, primals_767, buf606, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_722], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(buf606, primals_768, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf607, (4, 256, 1, 4, 4), (4096, 16, 16, 4, 1))
        buf608 = empty_strided_cuda((4, 256, 1, 4, 4), (4096, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_723, input_724], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32.run(buf607, primals_769, primals_770, primals_771, primals_772, buf608, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_725], Original ATen: [aten.convolution]
        buf609 = extern_kernels.convolution(buf608, primals_773, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf610 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_726], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf609, primals_774, primals_775, primals_776, primals_777, buf610, 8192, grid=grid(8192), stream=stream0)
        del primals_777
        # Topologically Sorted Source Nodes: [input_727], Original ATen: [aten.convolution]
        buf611 = extern_kernels.convolution(buf610, primals_778, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf611, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_729], Original ATen: [aten.convolution]
        buf612 = extern_kernels.convolution(buf598, primals_783, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf612, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf613 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_730, input_731], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf612, primals_784, primals_785, primals_786, primals_787, buf613, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_732], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf613, primals_788, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf614, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf615 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_733, input_734], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf614, primals_789, primals_790, primals_791, primals_792, buf615, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_735], Original ATen: [aten.convolution]
        buf616 = extern_kernels.convolution(buf615, primals_793, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf616, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf617 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf618 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [input_728, input_736, add_7, conv5_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf618, buf611, primals_779, primals_780, primals_781, primals_782, buf616, primals_794, primals_795, primals_796, primals_797, 32768, grid=grid(32768), stream=stream0)
        del primals_782
        del primals_797
        # Topologically Sorted Source Nodes: [input_737], Original ATen: [aten.convolution]
        buf619 = extern_kernels.convolution(buf618, primals_798, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf619, (4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_739], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf586, primals_803, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1))
        buf621 = empty_strided_cuda((4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_740, input_741], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35.run(buf620, primals_804, primals_805, primals_806, primals_807, buf621, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_742], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf621, primals_808, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf622, (4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1))
        buf623 = empty_strided_cuda((4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_743, input_744], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35.run(buf622, primals_809, primals_810, primals_811, primals_812, buf623, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_745], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf623, primals_813, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1))
        buf625 = empty_strided_cuda((4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1), torch.float32)
        buf626 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [input_738, input_746, add_8, conv6_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf626, buf619, primals_799, primals_800, primals_801, primals_802, buf624, primals_814, primals_815, primals_816, primals_817, 131072, grid=grid(131072), stream=stream0)
        del primals_802
        del primals_817
        # Topologically Sorted Source Nodes: [input_747], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(buf626, primals_818, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1))
        buf628 = empty_strided_cuda((4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_748, input_749], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35.run(buf627, primals_819, primals_820, primals_821, primals_822, buf628, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_750], Original ATen: [aten.convolution]
        buf629 = extern_kernels.convolution(buf628, primals_823, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf629, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf630 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_751, input_752], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_36.run(buf629, primals_824, primals_825, primals_826, primals_827, buf630, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_753], Original ATen: [aten.convolution]
        buf631 = extern_kernels.convolution(buf630, primals_828, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf631, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf632 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_754], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf631, primals_829, primals_830, primals_831, primals_832, buf632, 32768, grid=grid(32768), stream=stream0)
        del primals_832
        # Topologically Sorted Source Nodes: [input_755], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_833, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf634 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_756, input_757], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf633, primals_834, primals_835, primals_836, primals_837, buf634, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_758], Original ATen: [aten.convolution]
        buf635 = extern_kernels.convolution(buf634, primals_838, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf635, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf636 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_759, input_760], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf635, primals_839, primals_840, primals_841, primals_842, buf636, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_761], Original ATen: [aten.convolution]
        buf637 = extern_kernels.convolution(buf636, primals_843, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf637, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf638 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_762], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf637, primals_844, primals_845, primals_846, primals_847, buf638, 32768, grid=grid(32768), stream=stream0)
        del primals_847
        # Topologically Sorted Source Nodes: [input_763], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf638, primals_848, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf639, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf640 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_764, input_765], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf639, primals_849, primals_850, primals_851, primals_852, buf640, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_766], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, primals_853, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf641, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf642 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_767, input_768], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_30.run(buf641, primals_854, primals_855, primals_856, primals_857, buf642, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_769], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, primals_858, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf644 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_770], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf643, primals_859, primals_860, primals_861, primals_862, buf644, 8192, grid=grid(8192), stream=stream0)
        del primals_862
        # Topologically Sorted Source Nodes: [input_771], Original ATen: [aten.convolution]
        buf645 = extern_kernels.convolution(buf644, primals_863, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (4, 256, 1, 4, 4), (4096, 16, 16, 4, 1))
        buf646 = empty_strided_cuda((4, 256, 1, 4, 4), (4096, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_772, input_773], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32.run(buf645, primals_864, primals_865, primals_866, primals_867, buf646, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_774], Original ATen: [aten.convolution]
        buf647 = extern_kernels.convolution(buf646, primals_868, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=256, bias=None)
        assert_size_stride(buf647, (4, 256, 1, 4, 4), (4096, 16, 16, 4, 1))
        buf648 = empty_strided_cuda((4, 256, 1, 4, 4), (4096, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_775, input_776], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_32.run(buf647, primals_869, primals_870, primals_871, primals_872, buf648, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_777], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf648, primals_873, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (4, 128, 1, 4, 4), (2048, 16, 16, 4, 1))
        buf650 = empty_strided_cuda((4, 128, 1, 4, 4), (2048, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_778], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf649, primals_874, primals_875, primals_876, primals_877, buf650, 8192, grid=grid(8192), stream=stream0)
        del primals_877
        # Topologically Sorted Source Nodes: [input_779], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf650, primals_878, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf651, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_781], Original ATen: [aten.convolution]
        buf652 = extern_kernels.convolution(buf638, primals_883, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf652, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf653 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_782, input_783], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf652, primals_884, primals_885, primals_886, primals_887, buf653, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_784], Original ATen: [aten.convolution]
        buf654 = extern_kernels.convolution(buf653, primals_888, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=128, bias=None)
        assert_size_stride(buf654, (4, 128, 2, 8, 8), (16384, 128, 64, 8, 1))
        buf655 = empty_strided_cuda((4, 128, 2, 8, 8), (16384, 128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_785, input_786], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_38.run(buf654, primals_889, primals_890, primals_891, primals_892, buf655, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_787], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_893, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (4, 64, 2, 8, 8), (8192, 128, 64, 8, 1))
        buf657 = empty_strided_cuda((4, 64, 2, 8, 8), (8192, 128, 64, 8, 1), torch.float32)
        buf658 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [input_780, input_788, add_9, conv5_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf658, buf651, primals_879, primals_880, primals_881, primals_882, buf656, primals_894, primals_895, primals_896, primals_897, 32768, grid=grid(32768), stream=stream0)
        del primals_882
        del primals_897
        # Topologically Sorted Source Nodes: [input_789], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(buf658, primals_898, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf659, (4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_791], Original ATen: [aten.convolution]
        buf660 = extern_kernels.convolution(buf626, primals_903, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf660, (4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1))
        buf661 = empty_strided_cuda((4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_792, input_793], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35.run(buf660, primals_904, primals_905, primals_906, primals_907, buf661, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_794], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, primals_908, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=64, bias=None)
        assert_size_stride(buf662, (4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1))
        buf663 = empty_strided_cuda((4, 64, 4, 16, 16), (65536, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_795, input_796], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_35.run(buf662, primals_909, primals_910, primals_911, primals_912, buf663, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_797], Original ATen: [aten.convolution]
        buf664 = extern_kernels.convolution(buf663, primals_913, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf664, (4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1))
        buf665 = empty_strided_cuda((4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1), torch.float32)
        buf666 = buf665; del buf665  # reuse
        # Topologically Sorted Source Nodes: [input_790, input_798, add_10, conv6_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf666, buf659, primals_899, primals_900, primals_901, primals_902, buf664, primals_914, primals_915, primals_916, primals_917, 131072, grid=grid(131072), stream=stream0)
        del primals_902
        del primals_917
        # Topologically Sorted Source Nodes: [input_799], Original ATen: [aten.convolution]
        buf667 = extern_kernels.convolution(buf666, primals_918, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf667, (4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1))
        buf668 = empty_strided_cuda((4, 32, 4, 16, 16), (32768, 1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_800, input_801], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf667, primals_919, primals_920, primals_921, primals_922, buf668, 131072, grid=grid(131072), stream=stream0)
        del primals_922
        # Topologically Sorted Source Nodes: [input_802], Original ATen: [aten.convolution]
        buf669 = extern_kernels.convolution(buf668, primals_923, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf669, (4, 1, 4, 16, 16), (1024, 1024, 256, 16, 1))
        buf670 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_42.run(buf670, 4, grid=grid(4), stream=stream0)
        buf671 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_43.run(buf671, 4, grid=grid(4), stream=stream0)
        buf672 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf672, 64, grid=grid(64), stream=stream0)
        buf673 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_45.run(buf673, 64, grid=grid(64), stream=stream0)
        buf674 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf674, 64, grid=grid(64), stream=stream0)
        buf675 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_45.run(buf675, 64, grid=grid(64), stream=stream0)
        buf676 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_46.run(buf676, 64, grid=grid(64), stream=stream0)
        buf679 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_46.run(buf679, 64, grid=grid(64), stream=stream0)
        buf682 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_sub_47.run(buf682, 4, grid=grid(4), stream=stream0)
        buf677 = empty_strided_cuda((4, 1, 4, 64, 64), (16384, 65536, 4096, 64, 1), torch.float32)
        buf683 = buf677; del buf677  # reuse
        # Topologically Sorted Source Nodes: [cost3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_48.run(buf683, buf671, buf672, buf674, buf669, buf675, buf676, buf670, buf673, buf679, buf682, 65536, grid=grid(65536), stream=stream0)
        del buf669
        buf684 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pred3], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_49.run(buf683, buf684, 65536, grid=grid(65536), stream=stream0)
        buf685 = reinterpret_tensor(buf683, (4, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf683  # reuse
        # Topologically Sorted Source Nodes: [pred3], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_50.run(buf684, buf685, 65536, grid=grid(65536), stream=stream0)
        del buf684
        buf686 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [cost3, disp_values], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_51.run(buf686, 4, grid=grid(4), stream=stream0)
        buf687 = empty_strided_cuda((4, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, pred3_1], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_52.run(buf685, buf686, buf687, 16384, grid=grid(16384), stream=stream0)
    return (buf687, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_678, primals_679, primals_680, primals_681, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_698, primals_699, primals_700, primals_701, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_778, primals_779, primals_780, primals_781, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_798, primals_799, primals_800, primals_801, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_878, primals_879, primals_880, primals_881, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_898, primals_899, primals_900, primals_901, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_918, primals_919, primals_920, primals_921, primals_923, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf222, buf223, buf225, buf226, buf227, buf228, buf230, buf231, buf233, buf234, buf235, buf236, buf238, buf239, buf241, buf242, buf243, buf244, buf246, buf247, buf249, buf250, buf252, buf253, buf254, buf255, buf257, buf258, buf260, buf261, buf263, buf264, buf265, buf266, buf268, buf269, buf271, buf272, buf274, buf275, buf276, buf277, buf279, buf280, buf282, buf283, buf285, buf286, buf287, buf288, buf289, buf291, buf292, buf294, buf295, buf297, buf298, buf299, buf300, buf302, buf303, buf305, buf306, buf308, buf309, buf310, buf311, buf313, buf314, buf316, buf317, buf319, buf320, buf321, buf322, buf324, buf325, buf327, buf328, buf330, buf331, buf332, buf333, buf335, buf336, buf338, buf339, buf341, buf342, buf343, buf344, buf346, buf347, buf349, buf350, buf352, buf353, buf354, buf355, buf357, buf358, buf360, buf361, buf363, buf364, buf365, buf366, buf368, buf369, buf371, buf372, buf374, buf375, buf376, buf377, buf379, buf380, buf382, buf383, buf385, buf386, buf387, buf388, buf390, buf391, buf393, buf394, buf396, buf397, buf398, buf399, buf401, buf402, buf404, buf405, buf407, buf408, buf409, buf410, buf412, buf413, buf415, buf416, buf418, buf419, buf420, buf421, buf423, buf424, buf426, buf427, buf429, buf430, buf431, buf432, buf434, buf435, buf437, buf438, buf440, buf441, buf442, buf443, buf445, buf446, buf448, buf449, buf451, buf452, buf453, buf454, buf456, buf457, buf459, buf460, buf462, buf463, buf464, buf465, buf466, buf468, buf469, buf471, buf472, buf474, buf475, buf476, buf477, buf479, buf480, buf482, buf483, buf485, buf486, buf487, buf488, buf490, buf491, buf493, buf494, buf496, buf497, buf498, buf499, buf501, buf502, buf504, buf505, buf507, buf508, buf509, buf510, buf512, buf513, buf515, buf516, buf518, buf519, buf520, buf522, buf523, buf524, buf525, buf526, buf527, buf528, buf529, buf530, buf531, buf532, buf533, buf534, buf535, buf536, buf537, buf538, buf539, buf540, buf541, buf542, buf543, buf544, buf545, buf546, buf547, buf548, buf549, buf550, buf551, buf552, buf553, buf554, buf555, buf556, buf557, buf558, buf559, buf560, buf561, buf562, buf563, buf564, buf565, buf566, buf567, buf568, buf569, buf570, buf571, buf572, buf573, buf574, buf575, buf576, buf578, buf579, buf580, buf581, buf582, buf583, buf584, buf586, buf587, buf588, buf589, buf590, buf591, buf592, buf593, buf594, buf595, buf596, buf597, buf598, buf599, buf600, buf601, buf602, buf603, buf604, buf605, buf606, buf607, buf608, buf609, buf610, buf611, buf612, buf613, buf614, buf615, buf616, buf618, buf619, buf620, buf621, buf622, buf623, buf624, buf626, buf627, buf628, buf629, buf630, buf631, buf632, buf633, buf634, buf635, buf636, buf637, buf638, buf639, buf640, buf641, buf642, buf643, buf644, buf645, buf646, buf647, buf648, buf649, buf650, buf651, buf652, buf653, buf654, buf655, buf656, buf658, buf659, buf660, buf661, buf662, buf663, buf664, buf666, buf667, buf668, buf670, buf671, buf672, buf673, buf674, buf675, buf676, buf679, buf682, buf685, reinterpret_tensor(buf686, (1, 4, 1, 1), (4, 1, 1, 1), 0), buf688, buf689, buf690, buf691, buf692, buf693, buf694, buf695, buf696, buf697, buf698, buf699, buf700, buf701, buf702, buf703, buf704, buf705, buf706, buf707, buf708, buf709, buf710, buf711, buf712, buf713, buf714, buf715, buf716, buf717, buf718, buf719, buf720, buf721, buf722, buf723, buf724, buf725, buf726, buf727, buf728, buf729, buf730, buf731, buf732, buf733, buf734, buf735, buf736, buf737, buf738, buf739, buf740, buf741, buf742, buf743, buf744, buf745, buf746, buf747, buf748, buf749, buf750, buf751, buf752, buf753, buf754, buf755, buf756, buf757, buf758, buf759, buf760, buf761, buf762, buf763, buf764, buf765, buf766, buf767, buf768, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((9, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((9, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, 9, 1, 1), (9, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((96, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((96, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((120, 40, 1, 1, 1), (40, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((120, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((32, 120, 1, 1, 1), (120, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((96, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((96, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((32, 96, 1, 1, 1), (96, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((96, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((96, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((32, 96, 1, 1, 1), (96, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((96, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((96, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((32, 96, 1, 1, 1), (96, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((64, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((128, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((128, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((64, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((64, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((128, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((128, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((64, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((64, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((128, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((256, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((128, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((128, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((128, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((64, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((64, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((64, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((32, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((1, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
