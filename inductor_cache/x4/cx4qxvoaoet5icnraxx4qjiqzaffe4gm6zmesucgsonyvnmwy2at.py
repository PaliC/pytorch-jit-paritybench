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


# kernel path: inductor_cache/qg/cqgbotjqeeu72dldnqhbr6cnnofci4lig2xccorkeakhfdw36yts.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => clamp_max, clamp_min
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0.0), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
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
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t4/ct4zx7wi4c4flxqemjj2qvn6mntvvifpt7mvtntn3melooyfjn2x.py
# Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => clamp_max_1, clamp_min_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %primals_8, %primals_9, [2, 2], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4v/c4vyyiq7ty7k4wwiemlwwxxaelgdtjrocgm5eetnhnh3fxxqplca.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
# Source node to ATen node mapping:
#   x3 => clamp_min_4, convert_element_type_11, convert_element_type_8, iota, mul_12
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, 0), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_12, 0.0), kwargs = {})
#   %convert_element_type_11 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_2 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': (1,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/su/csuiher3qkoi4vqlu4twjqjvikutxroeq4vnkmn2wojcfw7uui6r.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   x3 => clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_8, iota, mul_12, sub_4
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, 0), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_12, 0.0), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_11), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_4, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_3 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': (1,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = 0.0
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/2f/c2f7zny46yenrwmmqkkzgbdfaud65so32hk3j4anwlackqdcu6sd.py
# Topologically Sorted Source Nodes: [input_10, input_11, input_12, x3, input_13, input_14, input_15, x2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_10 => convolution_3
#   input_11 => add_7, mul_10, mul_11, sub_3
#   input_12 => clamp_max_3, clamp_min_3
#   input_13 => convolution_4
#   input_14 => add_14, mul_18, mul_19, sub_9
#   input_15 => clamp_max_8, clamp_min_8
#   x2 => add_15
#   x3 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_10, add_11, add_12, mul_14, mul_15, mul_16, sub_5, sub_6, sub_8
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_2, %primals_20, %primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6.0), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_3, [None, None, %convert_element_type_9, %convert_element_type_11]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_3, [None, None, %convert_element_type_9, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_3, [None, None, %clamp_max_4, %convert_element_type_11]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_3, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %clamp_max_6), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_14), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_6), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_15), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %add_10), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %clamp_max_7), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_16), kwargs = {})
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_1, %primals_26, %primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_35), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_37), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_39), kwargs = {})
#   %clamp_min_8 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_14, 0.0), kwargs = {})
#   %clamp_max_8 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_8, 6.0), kwargs = {})
#   %add_15 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_8, %add_12), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_4(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr10 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp45 = tl.load(in_ptr11 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp51 = tl.load(in_ptr12 + (0))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp56 = tl.load(in_ptr13 + (0))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
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
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 + tmp6
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tmp9 / tmp29
    tmp31 = tmp30 * tmp11
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(tmp36, tmp18)
    tmp38 = triton_helpers.minimum(tmp37, tmp20)
    tmp41 = tl.full([XBLOCK], 1, tl.int32)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp40 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp40)
    tmp47 = tmp46 + tmp41
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tmp50 = tmp21 - tmp21
    tmp53 = tmp50 * tmp52
    tmp54 = tmp21 + tmp53
    tmp55 = tmp54 - tmp54
    tmp58 = tmp55 * tmp57
    tmp59 = tmp54 + tmp58
    tmp60 = tmp38 + tmp59
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp24, xmask)
    tl.store(in_out_ptr2 + (x2), tmp60, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bj/cbjorvxc3eqphtqizc2odhiskwmxlkakaqidhvhnonzjmr2fpo24.py
# Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x2_1 => convert_element_type_15
# Graph fragment:
#   %convert_element_type_15 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uh/cuhtdfvqja5t44jibacjgvwqhllkth7f5tgw4wxyqyukms4coyte.py
# Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   x2_1 => clamp_max_11, clamp_min_11, clamp_min_9, convert_element_type_14, iota_2, mul_20, sub_10
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, 0.0), kwargs = {})
#   %clamp_min_9 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_20, 0.0), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_9, %convert_element_type_17), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0.0), kwargs = {})
#   %clamp_max_11 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_6 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s3/cs3vs5p76o4xoxbpnmwm555thofvweocieeyqutjdg3s25qigwoz.py
# Topologically Sorted Source Nodes: [x2_1, input_16, input_17, input_18, x1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_16 => convolution_5
#   input_17 => add_22, mul_26, mul_27, sub_15
#   input_18 => clamp_max_13, clamp_min_13
#   x1 => add_23
#   x2_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_18, add_19, add_20, mul_22, mul_23, mul_24, sub_11, sub_12, sub_14
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_15, [None, None, %convert_element_type_15, %convert_element_type_17]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_15, [None, None, %convert_element_type_15, %clamp_max_10]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_15, [None, None, %clamp_max_9, %convert_element_type_17]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_15, [None, None, %clamp_max_9, %clamp_max_10]), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %clamp_max_11), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_22), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %clamp_max_11), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_23), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %add_18), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %clamp_max_12), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %mul_24), kwargs = {})
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %primals_32, %primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_43), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_45), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_47), kwargs = {})
#   %clamp_min_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_22, 0.0), kwargs = {})
#   %clamp_max_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_13, 6.0), kwargs = {})
#   %add_23 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_13, %add_20), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = xindex
    x1 = ((xindex // 4) % 4)
    x4 = ((xindex // 2) % 2)
    x3 = (xindex % 2)
    x5 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x6), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr11 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK], 1, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tmp9 = tmp8 + tmp4
    tmp10 = tmp8 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp8)
    tmp14 = tmp13 + tmp4
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tmp17 = tmp12 - tmp12
    tmp19 = tmp17 * tmp18
    tmp20 = tmp12 + tmp19
    tmp22 = tmp2 - tmp21
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.sqrt(tmp25)
    tmp27 = tl.full([1], 1, tl.int32)
    tmp28 = tmp27 / tmp26
    tmp29 = 1.0
    tmp30 = tmp28 * tmp29
    tmp31 = tmp22 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0
    tmp37 = triton_helpers.maximum(tmp35, tmp36)
    tmp38 = 6.0
    tmp39 = triton_helpers.minimum(tmp37, tmp38)
    tmp41 = tmp40 + tmp4
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp44 = tmp20 - tmp20
    tmp46 = tmp44 * tmp45
    tmp47 = tmp20 + tmp46
    tmp48 = tmp39 + tmp47
    tl.store(in_out_ptr0 + (x6), tmp2, xmask)
    tl.store(in_out_ptr1 + (x6), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lz/clz3wnzsiiicdw5qzyweqwoicegmpsdaaiyxoqmy5gof7n45hz34.py
# Topologically Sorted Source Nodes: [out1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out1 => convert_element_type_21
# Graph fragment:
#   %convert_element_type_21 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mg/cmgkt2xfwymnqchrf7po5xcsi7tusremib5aiprut4vecyjocecw.py
# Topologically Sorted Source Nodes: [out1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   out1 => add_24, clamp_max_14
# Graph fragment:
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_21, 1), kwargs = {})
#   %clamp_max_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_24, 1), kwargs = {})
triton_poi_fused_add_clamp_9 = async_compile.triton('triton_poi_fused_add_clamp_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.minimum(tmp8, tmp7)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yb/cybze5mghmgjrnsr2ymyqnsnfwq4dqq3gzkscny7ptjkalqd6ycn.py
# Topologically Sorted Source Nodes: [out1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   out1 => clamp_max_16, clamp_min_14, clamp_min_16, convert_element_type_20, iota_4, mul_28, sub_16
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_20 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, 0.3333333333333333), kwargs = {})
#   %clamp_min_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_28, 0.0), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_14, %convert_element_type_23), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_16, 0.0), kwargs = {})
#   %clamp_max_16 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_10 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
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


# kernel path: inductor_cache/lt/cltnfohd4bxuff2cvikcyhen73kk6y73et42zthcvovbptkivjna.py
# Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   input_22 => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%primals_1, [-1, -2], True), kwargs = {})
triton_per_fused_mean_11 = async_compile.triton('triton_per_fused_mean_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_11(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ll/cll4d3clcsh7ekyxabzdoqiixg6jjt7ufbg5arjn55c7ugt3sdcc.py
# Topologically Sorted Source Nodes: [out3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out3 => convert_element_type_27
# Graph fragment:
#   %convert_element_type_27 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ww/cwwz4xn3ubqhwnc24sqi5yyp2qzi3ydho5lcmqmmpuw4anc4gak7.py
# Topologically Sorted Source Nodes: [out1, out3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   out1 => convert_element_type_20, iota_4
#   out3 => clamp_max_21, clamp_min_19, clamp_min_21, mul_36, sub_22
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_20 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, 0.0), kwargs = {})
#   %clamp_min_19 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_36, 0.0), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_19, %convert_element_type_29), kwargs = {})
#   %clamp_min_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_22, 0.0), kwargs = {})
#   %clamp_max_21 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_21, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_13 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yl/cylpdmhzj6xziqno3hwn47kdkkvngov6eiphceqixypcr7krrelg.py
# Topologically Sorted Source Nodes: [out1, input_19, input_20, input_21, input_23, out3, mul, add_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   add_2 => add_36
#   input_19 => convolution_6
#   input_20 => add_30, mul_34, mul_35, sub_21
#   input_21 => clamp_max_18, clamp_min_18
#   input_23 => convolution_7
#   mul => mul_41
#   out1 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_26, add_27, add_28, mul_30, mul_31, mul_32, sub_17, sub_18, sub_20
#   out3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_33, add_34, add_35, mul_38, mul_39, mul_40, sub_23, sub_24, sub_26
# Graph fragment:
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_23, [None, None, %convert_element_type_21, %convert_element_type_23]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_23, [None, None, %convert_element_type_21, %clamp_max_15]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_23, [None, None, %clamp_max_14, %convert_element_type_23]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_23, [None, None, %clamp_max_14, %clamp_max_15]), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %clamp_max_16), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_30), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %clamp_max_16), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_31), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_27, %add_26), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %clamp_max_17), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %mul_32), kwargs = {})
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_38, %primals_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_51), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_53), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_55), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_30, 0.0), kwargs = {})
#   %clamp_max_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6.0), kwargs = {})
#   %convolution_7 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_44, %primals_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_7, [None, None, %convert_element_type_27, %convert_element_type_29]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_7, [None, None, %convert_element_type_27, %clamp_max_20]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_7, [None, None, %clamp_max_19, %convert_element_type_29]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_7, [None, None, %clamp_max_19, %clamp_max_20]), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %clamp_max_21), kwargs = {})
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_38), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %clamp_max_21), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_39), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %add_33), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %clamp_max_22), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %mul_40), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %clamp_max_18), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %add_35), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_14(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x6 = xindex
    x4 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_out_ptr1 + (x6), xmask)
    tmp33 = tl.load(in_ptr7 + (x4), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr11 + (x4), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr13 + (x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x4), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr15 + (x4), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr16 + (x4), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr17 + (x4), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 2*tmp22 + 4*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 2*tmp22 + 4*x2), xmask, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp34 = tmp32 + tmp33
    tmp36 = tl.full([XBLOCK], 1, tl.int32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tmp41 = tmp40 + tmp36
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp46 = tmp44 + tmp45
    tmp48 = tmp47 + tmp36
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tmp51 = tmp46 - tmp46
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 + tmp53
    tmp56 = tmp34 - tmp55
    tmp58 = 1e-05
    tmp59 = tmp57 + tmp58
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tl.full([1], 1, tl.int32)
    tmp62 = tmp61 / tmp60
    tmp63 = 1.0
    tmp64 = tmp62 * tmp63
    tmp65 = tmp56 * tmp64
    tmp67 = tmp65 * tmp66
    tmp69 = tmp67 + tmp68
    tmp70 = 0.0
    tmp71 = triton_helpers.maximum(tmp69, tmp70)
    tmp72 = 6.0
    tmp73 = triton_helpers.minimum(tmp71, tmp72)
    tmp74 = tmp31 * tmp73
    tmp76 = tmp75 + tmp36
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tmp79 = tmp54 - tmp54
    tmp81 = tmp79 * tmp80
    tmp82 = tmp54 + tmp81
    tmp83 = tmp74 + tmp82
    tl.store(in_out_ptr0 + (x6), tmp31, xmask)
    tl.store(in_out_ptr1 + (x6), tmp34, xmask)
    tl.store(in_out_ptr2 + (x6), tmp83, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 5, 5), (100, 25, 5, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 7, 7), (196, 49, 7, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, ), (1, ))
    assert_size_stride(primals_38, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (4, ), (1, ))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, ), (1, ))
    assert_size_stride(primals_43, (4, ), (1, ))
    assert_size_stride(primals_44, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_45, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 2, 2), (16, 4, 2, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_0.run(buf1, primals_3, primals_4, primals_5, primals_6, primals_7, buf2, 64, grid=grid(64), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 1, 1), (4, 1, 1, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_1.run(buf4, primals_9, primals_10, primals_11, primals_12, primals_13, buf5, 16, grid=grid(16), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_14, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 1, 1), (4, 1, 1, 1))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_1.run(buf7, primals_15, primals_16, primals_17, primals_18, primals_19, buf8, 16, grid=grid(16), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 4, 1, 1), (4, 1, 1, 1))
        buf14 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_2.run(buf14, 1, grid=grid(1), stream=stream0)
        buf15 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_2.run(buf15, 1, grid=grid(1), stream=stream0)
        buf16 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_3.run(buf16, 1, grid=grid(1), stream=stream0)
        buf17 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_3.run(buf17, 1, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf5, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 4, 1, 1), (4, 1, 1, 1))
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf19 = buf18; del buf18  # reuse
        buf20 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12, x3, input_13, input_14, input_15, x2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_4.run(buf10, buf19, buf20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, buf14, buf15, buf16, buf17, 16, grid=grid(16), stream=stream0)
        del primals_21
        del primals_27
        buf12 = empty_strided_cuda((1, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_2.run(buf12, 1, grid=grid(1), stream=stream0)
        buf13 = empty_strided_cuda((1, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_2.run(buf13, 1, grid=grid(1), stream=stream0)
        buf21 = empty_strided_cuda((2, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf21, 2, grid=grid(2), stream=stream0)
        buf22 = empty_strided_cuda((2, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf22, 2, grid=grid(2), stream=stream0)
        buf23 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf23, 2, grid=grid(2), stream=stream0)
        buf24 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf24, 2, grid=grid(2), stream=stream0)
        buf25 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_6.run(buf25, 2, grid=grid(2), stream=stream0)
        buf27 = empty_strided_cuda((2, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x2_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_6.run(buf27, 2, grid=grid(2), stream=stream0)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf2, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 4, 2, 2), (16, 4, 2, 1))
        buf29 = buf28; del buf28  # reuse
        buf26 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf30 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x2_1, input_16, input_17, input_18, x1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_7.run(buf29, buf30, primals_33, buf21, buf23, buf20, buf24, buf25, primals_34, primals_35, primals_36, primals_37, buf22, buf27, 64, grid=grid(64), stream=stream0)
        del primals_33
        buf31 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(buf31, 4, grid=grid(4), stream=stream0)
        buf32 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_9.run(buf32, 4, grid=grid(4), stream=stream0)
        buf33 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(buf33, 4, grid=grid(4), stream=stream0)
        buf34 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_9.run(buf34, 4, grid=grid(4), stream=stream0)
        buf35 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_10.run(buf35, 4, grid=grid(4), stream=stream0)
        buf37 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_10.run(buf37, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(primals_1, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 4, 4, 4), (64, 16, 4, 1))
        buf41 = buf20; del buf20  # reuse
        buf42 = reinterpret_tensor(buf41, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_11.run(buf42, primals_1, 16, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 4, 1, 1), (4, 1, 1, 1))
        buf44 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf44, 4, grid=grid(4), stream=stream0)
        buf45 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf45, 4, grid=grid(4), stream=stream0)
        buf46 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out1, out3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf46, 4, grid=grid(4), stream=stream0)
        buf47 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf47, 4, grid=grid(4), stream=stream0)
        buf48 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out1, out3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_13.run(buf48, 4, grid=grid(4), stream=stream0)
        buf50 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_13.run(buf50, 4, grid=grid(4), stream=stream0)
        buf36 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf38 = buf36; del buf36  # reuse
        buf40 = buf39; del buf39  # reuse
        buf49 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf51 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [out1, input_19, input_20, input_21, input_23, out3, mul, add_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_14.run(buf38, buf40, buf51, buf31, buf33, buf30, buf34, buf35, buf32, buf37, primals_39, buf44, buf46, buf43, primals_45, buf47, buf48, primals_40, primals_41, primals_42, primals_43, buf45, buf50, 256, grid=grid(256), stream=stream0)
        del buf30
        del buf43
        del primals_39
        del primals_45
    return (buf51, primals_1, primals_2, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_32, primals_34, primals_35, primals_36, primals_37, primals_38, primals_40, primals_41, primals_42, primals_43, primals_44, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf12, buf13, buf14, buf15, buf16, buf17, buf19, buf21, buf22, buf23, buf24, buf25, buf27, buf29, buf31, buf32, buf33, buf34, buf35, buf37, buf38, buf40, buf42, buf44, buf45, buf46, buf47, buf48, buf50, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 5, 5), (100, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 7, 7), (196, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
