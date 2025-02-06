# AOT ID: ['4_forward']
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


# kernel path: inductor_cache/lw/clwzjxqatem4342fqqnlqeiv4w4iltzckzrolklzcwr5driop4pm.py
# Topologically Sorted Source Nodes: [x_1, softplus, tanh, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus => exp, gt, log1p, where
#   tanh => tanh
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => mul_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %tanh), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
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
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f3/cf3f4h54hgsommxi4svepai5pyuprusdoh5b4r4lsly7comeh6v4.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_4 => add_3, mul_5, mul_6, sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
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


# kernel path: inductor_cache/kp/ckpiagj52wkinkcairlrlw7bz7bjcdod77rc6txzs6sku6ppnolz.py
# Topologically Sorted Source Nodes: [x_10, softplus_3, tanh_3, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
# Source node to ATen node mapping:
#   softplus_3 => exp_3, gt_3, log1p_3, where_3
#   tanh_3 => tanh_3
#   x_10 => add_7, mul_13, mul_14, sub_3
#   x_11 => mul_15
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_7,), kwargs = {})
#   %log1p_3 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_3,), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 20), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_7, %log1p_3), kwargs = {})
#   %tanh_3 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_3,), kwargs = {})
#   %mul_15 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %tanh_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 2)
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
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rg/crgjopjmfgnfep2np624242dabgfdn52xpa3fct6yzwjxcfatfsq.py
# Topologically Sorted Source Nodes: [x_13, softplus_4, tanh_4, x_14, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_1 => add_10
#   softplus_4 => exp_4, gt_4, log1p_4, where_4
#   tanh_4 => tanh_4
#   x_13 => add_9, mul_17, mul_18, sub_4
#   x_14 => mul_19
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %exp_4 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_9,), kwargs = {})
#   %log1p_4 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_4,), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 20), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_9, %log1p_4), kwargs = {})
#   %tanh_4 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_4,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %tanh_4), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
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
    tmp17 = 20.0
    tmp18 = tmp15 > tmp17
    tmp19 = tl_math.exp(tmp15)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tl.where(tmp18, tmp15, tmp20)
    tmp22 = libdevice.tanh(tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/74/c74gly5qg7456yumx4y2ygbbmn452ycipzdgc76rgpd2aq7i6u3e.py
# Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_18 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_23, %mul_7], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 8)
    x0 = (xindex % 4)
    x2 = xindex // 32
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 16*x2), tmp4 & xmask, other=0.0)
    tmp6 = 20.0
    tmp7 = tmp5 > tmp6
    tmp8 = tl_math.exp(tmp5)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 8, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr1 + (x0 + 4*((-4) + x1) + 16*x2), tmp15 & xmask, other=0.0)
    tmp19 = 20.0
    tmp20 = tmp18 > tmp19
    tmp21 = tl_math.exp(tmp18)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp14, tmp27)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_18, (2, ), (1, ))
    assert_size_stride(primals_19, (2, ), (1, ))
    assert_size_stride(primals_20, (2, ), (1, ))
    assert_size_stride(primals_21, (2, ), (1, ))
    assert_size_stride(primals_22, (4, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 2, 2), (16, 4, 2, 1))
        buf1 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1, softplus, tanh, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_0.run(buf2, buf0, primals_3, primals_4, primals_5, primals_6, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 2, 2), (16, 4, 2, 1))
        buf4 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_1.run(buf3, primals_8, primals_9, primals_10, primals_11, buf4, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf2, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 4, 2, 2), (16, 4, 2, 1))
        buf6 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_7, softplus_2, tanh_2, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_0.run(buf7, buf5, primals_13, primals_14, primals_15, primals_16, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 2, 2, 2), (8, 4, 2, 1))
        buf9 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_10, softplus_3, tanh_3, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_2.run(buf10, buf8, primals_18, primals_19, primals_20, primals_21, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 4, 2, 2), (16, 4, 2, 1))
        buf12 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_13, softplus_4, tanh_4, x_14, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_softplus_tanh_3.run(buf13, buf11, primals_23, primals_24, primals_25, primals_26, buf7, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 4, 2, 2), (16, 4, 2, 1))
        buf15 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_1.run(buf14, primals_28, primals_29, primals_30, primals_31, buf15, 64, grid=grid(64), stream=stream0)
        buf16 = empty_strided_cuda((4, 8, 2, 2), (32, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf15, buf4, buf16, 128, grid=grid(128), stream=stream0)
        del buf15
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 4, 2, 2), (16, 4, 2, 1))
        buf18 = buf4; del buf4  # reuse
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_20, softplus_6, tanh_6, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.softplus, aten.tanh, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_softplus_tanh_0.run(buf19, buf17, primals_33, primals_34, primals_35, primals_36, 64, grid=grid(64), stream=stream0)
    return (buf19, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, buf0, buf2, buf3, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
