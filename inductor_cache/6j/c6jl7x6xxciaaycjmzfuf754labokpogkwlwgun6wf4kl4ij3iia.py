# AOT ID: ['45_forward']
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


# kernel path: inductor_cache/oc/cocgqe627hqesvtla5o7te3sovt62uofwplksrnopx5fhaeikbah.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_1 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%primals_1,), kwargs = {})
triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ln/clnfg5lexj3aegtuuz2zrisctrahrfrlkjmmnk3f2w24vz35wff3.py
# Topologically Sorted Source Nodes: [input_3, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_3 => add_1, mul_1, mul_2, sub
#   x_8 => relu_4
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu_4 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f7/cf7wnyt56katzgfna6fgbhp4ycumbmgkum4bi7sgoiebouial2ka.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_3 => add_5, mul_7, mul_8, sub_2
#   x_4 => relu_3
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jc/cjcqjzlz25l35t443lwdd4ijcnw3pzp2usv2mi4lfv5hswwwshcl.py
# Topologically Sorted Source Nodes: [x_7, x_15, x_comb_iter_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_15 => add_11, mul_16, mul_17, sub_5
#   x_7 => add_7, mul_10, mul_11, sub_3
#   x_comb_iter_0 => add_12
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %add_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    x2 = xindex // 64
    x4 = (xindex % 64)
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
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (x4 + 384*x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yd/cyd2mwxe7egr3renivsfv6huiim3h3ahfw42hyt3saapidgdv3ak.py
# Topologically Sorted Source Nodes: [x_comb_iter_2_left, x_comb_iter_2, x_comb_iter_3_left, x_comb_iter_3, x_39, x_comb_iter_4, x_out], Original ATen: [aten.avg_pool2d, aten.add, aten._native_batch_norm_legit_no_training, aten.cat]
# Source node to ATen node mapping:
#   x_39 => add_27, mul_34, mul_35, sub_11
#   x_comb_iter_2 => add_22
#   x_comb_iter_2_left => avg_pool2d
#   x_comb_iter_3 => add_23
#   x_comb_iter_3_left => avg_pool2d_1
#   x_comb_iter_4 => add_28
#   x_out => cat
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_3, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%avg_pool2d, %add_1), kwargs = {})
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%add_1, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%avg_pool2d_1, %avg_pool2d_1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %add_3), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_12, %add_21, %add_22, %add_23, %add_28], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_cat_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x8 = xindex
    x3 = (xindex % 64)
    x4 = xindex // 64
    x6 = ((xindex // 16) % 4)
    tmp72 = tl.load(in_ptr1 + (x8), xmask)
    tmp75 = tl.load(in_ptr2 + (x8), xmask)
    tmp76 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr6 + (x6), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr0 + (x8), xmask)
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + x8), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + x8), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + x8), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x8), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x8), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x8), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + x8), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + x8), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + x8), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))) + ((4) * ((4) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (4)))*((4) * ((4) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (4))) + ((-1)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((4) * ((4) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (4)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((4) * ((4) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (4))))
    tmp53 = tmp51 / tmp52
    tmp54 = tl.load(in_ptr1 + ((-5) + x8), tmp10 & xmask, other=0.0)
    tmp55 = tl.load(in_ptr1 + ((-4) + x8), tmp16 & xmask, other=0.0)
    tmp56 = tmp55 + tmp54
    tmp57 = tl.load(in_ptr1 + ((-3) + x8), tmp23 & xmask, other=0.0)
    tmp58 = tmp57 + tmp56
    tmp59 = tl.load(in_ptr1 + ((-1) + x8), tmp30 & xmask, other=0.0)
    tmp60 = tmp59 + tmp58
    tmp61 = tl.load(in_ptr1 + (x8), tmp33 & xmask, other=0.0)
    tmp62 = tmp61 + tmp60
    tmp63 = tl.load(in_ptr1 + (1 + x8), tmp36 & xmask, other=0.0)
    tmp64 = tmp63 + tmp62
    tmp65 = tl.load(in_ptr1 + (3 + x8), tmp43 & xmask, other=0.0)
    tmp66 = tmp65 + tmp64
    tmp67 = tl.load(in_ptr1 + (4 + x8), tmp46 & xmask, other=0.0)
    tmp68 = tmp67 + tmp66
    tmp69 = tl.load(in_ptr1 + (5 + x8), tmp49 & xmask, other=0.0)
    tmp70 = tmp69 + tmp68
    tmp71 = tmp70 / tmp52
    tmp73 = tmp53 + tmp72
    tmp74 = tmp71 + tmp71
    tmp77 = tmp75 - tmp76
    tmp79 = 0.001
    tmp80 = tmp78 + tmp79
    tmp81 = libdevice.sqrt(tmp80)
    tmp82 = tl.full([1], 1, tl.int32)
    tmp83 = tmp82 / tmp81
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp77 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp92 = tmp90 + tmp91
    tl.store(out_ptr2 + (x3 + 384*x4), tmp72, xmask)
    tl.store(out_ptr3 + (x3 + 384*x4), tmp73, xmask)
    tl.store(out_ptr4 + (x3 + 384*x4), tmp74, xmask)
    tl.store(out_ptr5 + (x3 + 384*x4), tmp92, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_8, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_14, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_20, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_26, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_38, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (4, ), (1, ))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, ), (1, ))
    assert_size_stride(primals_43, (4, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_44, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_50, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_51, (4, ), (1, ))
    assert_size_stride(primals_52, (4, ), (1, ))
    assert_size_stride(primals_53, (4, ), (1, ))
    assert_size_stride(primals_54, (4, ), (1, ))
    assert_size_stride(primals_55, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_57, (4, ), (1, ))
    assert_size_stride(primals_58, (4, ), (1, ))
    assert_size_stride(primals_59, (4, ), (1, ))
    assert_size_stride(primals_60, (4, ), (1, ))
    assert_size_stride(primals_61, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_62, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (4, ), (1, ))
    assert_size_stride(primals_65, (4, ), (1, ))
    assert_size_stride(primals_66, (4, ), (1, ))
    assert_size_stride(primals_67, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_69, (4, ), (1, ))
    assert_size_stride(primals_70, (4, ), (1, ))
    assert_size_stride(primals_71, (4, ), (1, ))
    assert_size_stride(primals_72, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(primals_1, buf0, 256, grid=grid(256), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 4, 4), (64, 16, 4, 1))
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf1, primals_3, primals_4, primals_5, primals_6, buf2, buf12, 256, grid=grid(256), stream=stream0)
        del primals_6
        buf3 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(primals_7, buf3, 256, grid=grid(256), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 4, 4), (64, 16, 4, 1))
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf4, primals_9, primals_10, primals_11, primals_12, buf5, buf6, 256, grid=grid(256), stream=stream0)
        del primals_12
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_13, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf7, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 4, 4, 4), (64, 16, 4, 1))
        buf9 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf8, primals_15, primals_16, primals_17, primals_18, buf9, 256, grid=grid(256), stream=stream0)
        del primals_18
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_19, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf10, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf13, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 4, 4, 4), (64, 16, 4, 1))
        buf15 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf14, primals_27, primals_28, primals_29, primals_30, buf15, 256, grid=grid(256), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf16, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 4, 4, 4), (64, 16, 4, 1))
        buf41 = empty_strided_cuda((4, 24, 4, 4), (384, 16, 4, 1), torch.float32)
        buf18 = reinterpret_tensor(buf41, (4, 4, 4, 4), (384, 16, 4, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [x_7, x_15, x_comb_iter_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf11, primals_21, primals_22, primals_23, primals_24, buf17, primals_33, primals_34, primals_35, primals_36, buf18, 256, grid=grid(256), stream=stream0)
        del primals_24
        del primals_36
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf12, primals_37, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf19, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 4, 4, 4), (64, 16, 4, 1))
        buf21 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf20, primals_39, primals_40, primals_41, primals_42, buf21, 256, grid=grid(256), stream=stream0)
        del primals_42
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_43, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf22, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf12, primals_49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf24, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 4, 4, 4), (64, 16, 4, 1))
        buf26 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf25, primals_51, primals_52, primals_53, primals_54, buf26, 256, grid=grid(256), stream=stream0)
        del primals_54
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf27, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 4, 4, 4), (64, 16, 4, 1))
        buf29 = reinterpret_tensor(buf41, (4, 4, 4, 4), (384, 16, 4, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [x_23, x_31, x_comb_iter_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf23, primals_45, primals_46, primals_47, primals_48, buf28, primals_57, primals_58, primals_59, primals_60, buf29, 256, grid=grid(256), stream=stream0)
        del primals_48
        del primals_60
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf6, primals_61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf32, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 4, 4, 4), (64, 16, 4, 1))
        buf34 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_35, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf33, primals_63, primals_64, primals_65, primals_66, buf34, 256, grid=grid(256), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf35, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 4, 4, 4), (64, 16, 4, 1))
        buf37 = reinterpret_tensor(buf41, (4, 4, 4, 4), (384, 16, 4, 1), 0)  # alias
        buf38 = reinterpret_tensor(buf41, (4, 4, 4, 4), (384, 16, 4, 1), 192)  # alias
        buf39 = reinterpret_tensor(buf41, (4, 4, 4, 4), (384, 16, 4, 1), 256)  # alias
        buf40 = reinterpret_tensor(buf41, (4, 4, 4, 4), (384, 16, 4, 1), 320)  # alias
        # Topologically Sorted Source Nodes: [x_comb_iter_2_left, x_comb_iter_2, x_comb_iter_3_left, x_comb_iter_3, x_39, x_comb_iter_4, x_out], Original ATen: [aten.avg_pool2d, aten.add, aten._native_batch_norm_legit_no_training, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_cat_4.run(buf5, buf2, buf36, primals_69, primals_70, primals_71, primals_72, buf37, buf38, buf39, buf40, 256, grid=grid(256), stream=stream0)
        del primals_72
    return (buf41, primals_2, primals_3, primals_4, primals_5, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_19, primals_20, primals_21, primals_22, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf32, buf33, buf34, buf35, buf36, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
