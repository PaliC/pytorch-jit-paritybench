# AOT ID: ['64_forward']
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


# kernel path: inductor_cache/yo/cyoiut4hkpv467xhjdfkc22qvbhgp3bf3klob4w2jdx72stw4yv2.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => clamp_max, clamp_min
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_8), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_11), kwargs = {})
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
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 8)
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4o2fpgpv4pp4ibdarugv2vmdwfhf6g74mbzk6fkb4jud7pbggg.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => clamp_max_1, clamp_min_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_14), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_17), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_20), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_23), kwargs = {})
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 8) % 8)
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cf/ccfkpckppb4cwlafpf5rjzxiiy34uiuntbjf7u3wvknppssei5gw.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_7, mul_8, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_26), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_29), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_32), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_35), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 8) % 8)
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


# kernel path: inductor_cache/7z/c7ztxvieznilmypiyxl5m7glny5xdc6xnuv2o3trfkqmlm7lz72w.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_10 => add_7, mul_10, mul_11, sub_3
#   input_11 => clamp_max_2, clamp_min_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_38), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_41), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_44), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_47), kwargs = {})
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
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 8) % 16)
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wl/cwldxkzvncfa7peymidyttwfjnjfz6tuaccrf5x6jcxyds426ge5.py
# Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_21 => add_15, mul_22, mul_23, sub_7
#   input_22 => clamp_max_5, clamp_min_5
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_86), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_89), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_92), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_95), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_15, 0.0), kwargs = {})
#   %clamp_max_5 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g2/cg2azx7eymxzpm7rbx3i7ou4p6pchidu6tivdz4smxpldl4i54aj.py
# Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_24 => add_17, mul_25, mul_26, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_98), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_101), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_104), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_107), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fl/cflfriavddynjpcpzfmgs5dc5ccinzhk3bf2jzysdia2spfzhxfp.py
# Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_26 => add_19, mul_28, mul_29, sub_9
#   input_27 => clamp_max_6, clamp_min_6
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_110), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_113), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_116), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_119), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_19, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f5/cf5t6stbuwjnfjg56ccf2zxxqgcrhys2xmykpeb74tgyoi44aqfo.py
# Topologically Sorted Source Nodes: [input_34, input_42, add, conv5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add => add_32
#   conv5 => relu
#   input_34 => add_25, mul_37, mul_38, sub_12
#   input_42 => add_31, mul_46, mul_47, sub_15
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_146), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_149), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_152), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_155), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_182), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_185), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_188), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_191), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %add_31), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_32,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 8) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/o5/co5t6d4jba2idnxuln7gcdgco7jkzaa55jhobxvwcn5c34drl3lh.py
# Topologically Sorted Source Nodes: [input_44, input_52, add_1, conv6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   add_1 => add_41
#   conv6 => relu_1
#   input_44 => add_34, mul_49, mul_50, sub_16
#   input_52 => add_40, mul_58, mul_59, sub_19
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_194), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_197), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_200), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_203), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_230), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_233), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_236), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_239), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %add_40), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), xmask)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
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
    tmp32 = 0.0
    tmp33 = tmp31 <= tmp32
    tl.store(in_out_ptr0 + (x3), tmp31, xmask)
    tl.store(out_ptr0 + (x3), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101 = args
    args.clear()
    assert_size_stride(primals_1, (8, 4, 1, 1, 1), (4, 1, 1, 1, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1))
    assert_size_stride(primals_3, (8, ), (1, ))
    assert_size_stride(primals_4, (8, ), (1, ))
    assert_size_stride(primals_5, (8, ), (1, ))
    assert_size_stride(primals_6, (8, ), (1, ))
    assert_size_stride(primals_7, (8, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_8, (8, ), (1, ))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, ), (1, ))
    assert_size_stride(primals_12, (8, 8, 1, 1, 1), (8, 1, 1, 1, 1))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (8, ), (1, ))
    assert_size_stride(primals_15, (8, ), (1, ))
    assert_size_stride(primals_16, (8, ), (1, ))
    assert_size_stride(primals_17, (16, 8, 1, 1, 1), (8, 1, 1, 1, 1))
    assert_size_stride(primals_18, (16, ), (1, ))
    assert_size_stride(primals_19, (16, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (16, ), (1, ))
    assert_size_stride(primals_22, (16, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_23, (16, ), (1, ))
    assert_size_stride(primals_24, (16, ), (1, ))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (16, ), (1, ))
    assert_size_stride(primals_27, (8, 16, 1, 1, 1), (16, 1, 1, 1, 1))
    assert_size_stride(primals_28, (8, ), (1, ))
    assert_size_stride(primals_29, (8, ), (1, ))
    assert_size_stride(primals_30, (8, ), (1, ))
    assert_size_stride(primals_31, (8, ), (1, ))
    assert_size_stride(primals_32, (16, 8, 1, 1, 1), (8, 1, 1, 1, 1))
    assert_size_stride(primals_33, (16, ), (1, ))
    assert_size_stride(primals_34, (16, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (16, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_38, (16, ), (1, ))
    assert_size_stride(primals_39, (16, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (16, ), (1, ))
    assert_size_stride(primals_42, (16, 16, 1, 1, 1), (16, 1, 1, 1, 1))
    assert_size_stride(primals_43, (16, ), (1, ))
    assert_size_stride(primals_44, (16, ), (1, ))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (16, ), (1, ))
    assert_size_stride(primals_47, (32, 16, 1, 1, 1), (16, 1, 1, 1, 1))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (32, ), (1, ))
    assert_size_stride(primals_50, (32, ), (1, ))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (32, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_53, (32, ), (1, ))
    assert_size_stride(primals_54, (32, ), (1, ))
    assert_size_stride(primals_55, (32, ), (1, ))
    assert_size_stride(primals_56, (32, ), (1, ))
    assert_size_stride(primals_57, (16, 32, 1, 1, 1), (32, 1, 1, 1, 1))
    assert_size_stride(primals_58, (16, ), (1, ))
    assert_size_stride(primals_59, (16, ), (1, ))
    assert_size_stride(primals_60, (16, ), (1, ))
    assert_size_stride(primals_61, (16, ), (1, ))
    assert_size_stride(primals_62, (16, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_63, (8, ), (1, ))
    assert_size_stride(primals_64, (8, ), (1, ))
    assert_size_stride(primals_65, (8, ), (1, ))
    assert_size_stride(primals_66, (8, ), (1, ))
    assert_size_stride(primals_67, (16, 8, 1, 1, 1), (8, 1, 1, 1, 1))
    assert_size_stride(primals_68, (16, ), (1, ))
    assert_size_stride(primals_69, (16, ), (1, ))
    assert_size_stride(primals_70, (16, ), (1, ))
    assert_size_stride(primals_71, (16, ), (1, ))
    assert_size_stride(primals_72, (16, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_73, (16, ), (1, ))
    assert_size_stride(primals_74, (16, ), (1, ))
    assert_size_stride(primals_75, (16, ), (1, ))
    assert_size_stride(primals_76, (16, ), (1, ))
    assert_size_stride(primals_77, (8, 16, 1, 1, 1), (16, 1, 1, 1, 1))
    assert_size_stride(primals_78, (8, ), (1, ))
    assert_size_stride(primals_79, (8, ), (1, ))
    assert_size_stride(primals_80, (8, ), (1, ))
    assert_size_stride(primals_81, (8, ), (1, ))
    assert_size_stride(primals_82, (8, 4, 3, 3, 3), (108, 27, 9, 3, 1))
    assert_size_stride(primals_83, (4, ), (1, ))
    assert_size_stride(primals_84, (4, ), (1, ))
    assert_size_stride(primals_85, (4, ), (1, ))
    assert_size_stride(primals_86, (4, ), (1, ))
    assert_size_stride(primals_87, (8, 4, 1, 1, 1), (4, 1, 1, 1, 1))
    assert_size_stride(primals_88, (8, ), (1, ))
    assert_size_stride(primals_89, (8, ), (1, ))
    assert_size_stride(primals_90, (8, ), (1, ))
    assert_size_stride(primals_91, (8, ), (1, ))
    assert_size_stride(primals_92, (8, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_93, (8, ), (1, ))
    assert_size_stride(primals_94, (8, ), (1, ))
    assert_size_stride(primals_95, (8, ), (1, ))
    assert_size_stride(primals_96, (8, ), (1, ))
    assert_size_stride(primals_97, (4, 8, 1, 1, 1), (8, 1, 1, 1, 1))
    assert_size_stride(primals_98, (4, ), (1, ))
    assert_size_stride(primals_99, (4, ), (1, ))
    assert_size_stride(primals_100, (4, ), (1, ))
    assert_size_stride(primals_101, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 8, 4, 4, 4), (512, 64, 16, 4, 1))
        buf1 = empty_strided_cuda((4, 8, 4, 4, 4), (512, 64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_7, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=8, bias=None)
        assert_size_stride(buf2, (4, 8, 2, 2, 2), (64, 8, 4, 2, 1))
        buf3 = empty_strided_cuda((4, 8, 2, 2, 2), (64, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_1.run(buf2, primals_8, primals_9, primals_10, primals_11, buf3, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 8, 2, 2, 2), (64, 8, 4, 2, 1))
        buf5 = empty_strided_cuda((4, 8, 2, 2, 2), (64, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf4, primals_13, primals_14, primals_15, primals_16, buf5, 256, grid=grid(256), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_17, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 16, 2, 2, 2), (128, 8, 4, 2, 1))
        buf7 = empty_strided_cuda((4, 16, 2, 2, 2), (128, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf6, primals_18, primals_19, primals_20, primals_21, buf7, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_22, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=16, bias=None)
        assert_size_stride(buf8, (4, 16, 2, 2, 2), (128, 8, 4, 2, 1))
        buf9 = empty_strided_cuda((4, 16, 2, 2, 2), (128, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf8, primals_23, primals_24, primals_25, primals_26, buf9, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_27, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 8, 2, 2, 2), (64, 8, 4, 2, 1))
        buf11 = empty_strided_cuda((4, 8, 2, 2, 2), (64, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf10, primals_28, primals_29, primals_30, primals_31, buf11, 256, grid=grid(256), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_32, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 16, 2, 2, 2), (128, 8, 4, 2, 1))
        buf13 = empty_strided_cuda((4, 16, 2, 2, 2), (128, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf12, primals_33, primals_34, primals_35, primals_36, buf13, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_37, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=16, bias=None)
        assert_size_stride(buf14, (4, 16, 1, 1, 1), (16, 1, 1, 1, 1))
        buf15 = empty_strided_cuda((4, 16, 1, 1, 1), (16, 1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4.run(buf14, primals_38, primals_39, primals_40, primals_41, buf15, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_42, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 16, 1, 1, 1), (16, 1, 1, 1, 1))
        buf17 = empty_strided_cuda((4, 16, 1, 1, 1), (16, 1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf16, primals_43, primals_44, primals_45, primals_46, buf17, 64, grid=grid(64), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_47, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 1, 1, 1), (32, 1, 1, 1, 1))
        buf19 = empty_strided_cuda((4, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6.run(buf18, primals_48, primals_49, primals_50, primals_51, buf19, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_52, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=32, bias=None)
        assert_size_stride(buf20, (4, 32, 1, 1, 1), (32, 1, 1, 1, 1))
        buf21 = empty_strided_cuda((4, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6.run(buf20, primals_53, primals_54, primals_55, primals_56, buf21, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_57, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 16, 1, 1, 1), (16, 1, 1, 1, 1))
        buf23 = empty_strided_cuda((4, 16, 1, 1, 1), (16, 1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf22, primals_58, primals_59, primals_60, primals_61, buf23, 64, grid=grid(64), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_62, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf24, (4, 8, 2, 2, 2), (64, 8, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf11, primals_67, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 16, 2, 2, 2), (128, 8, 4, 2, 1))
        buf26 = empty_strided_cuda((4, 16, 2, 2, 2), (128, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf25, primals_68, primals_69, primals_70, primals_71, buf26, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_72, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=16, bias=None)
        assert_size_stride(buf27, (4, 16, 2, 2, 2), (128, 8, 4, 2, 1))
        buf28 = empty_strided_cuda((4, 16, 2, 2, 2), (128, 8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf27, primals_73, primals_74, primals_75, primals_76, buf28, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_77, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 8, 2, 2, 2), (64, 8, 4, 2, 1))
        buf30 = empty_strided_cuda((4, 8, 2, 2, 2), (64, 8, 4, 2, 1), torch.float32)
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [input_34, input_42, add, conv5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf31, buf24, primals_63, primals_64, primals_65, primals_66, buf29, primals_78, primals_79, primals_80, primals_81, 256, grid=grid(256), stream=stream0)
        del primals_66
        del primals_81
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_82, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True, output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf32, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(primals_2, primals_87, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 8, 4, 4, 4), (512, 64, 16, 4, 1))
        buf34 = empty_strided_cuda((4, 8, 4, 4, 4), (512, 64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, input_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0.run(buf33, primals_88, primals_89, primals_90, primals_91, buf34, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_92, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=8, bias=None)
        assert_size_stride(buf35, (4, 8, 4, 4, 4), (512, 64, 16, 4, 1))
        buf36 = empty_strided_cuda((4, 8, 4, 4, 4), (512, 64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_0.run(buf35, primals_93, primals_94, primals_95, primals_96, buf36, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_97, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1))
        buf38 = empty_strided_cuda((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), torch.float32)
        buf39 = buf38; del buf38  # reuse
        buf40 = empty_strided_cuda((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_44, input_52, add_1, conv6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_8.run(buf39, buf32, primals_83, primals_84, primals_85, primals_86, buf37, primals_98, primals_99, primals_100, primals_101, buf40, 1024, grid=grid(1024), stream=stream0)
        del primals_101
        del primals_86
    return (buf39, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf40, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 4, 1, 1, 1), (4, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, 8, 1, 1, 1), (8, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, 8, 1, 1, 1), (8, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((8, 16, 1, 1, 1), (16, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((16, 8, 1, 1, 1), (8, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((16, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((16, 16, 1, 1, 1), (16, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, 16, 1, 1, 1), (16, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((32, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((16, 32, 1, 1, 1), (32, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((16, 8, 3, 3, 3), (216, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((16, 8, 1, 1, 1), (8, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((16, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((8, 16, 1, 1, 1), (16, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((8, 4, 3, 3, 3), (108, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((8, 4, 1, 1, 1), (4, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((8, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((4, 8, 1, 1, 1), (8, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
