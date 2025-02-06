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


# kernel path: inductor_cache/sw/cswqlarrndk6c6vehy3wfnp7vv42nswgx5xwfzmupfb6cgn42eet.py
# Topologically Sorted Source Nodes: [batch_norm, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   relu => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
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
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 24)
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


# kernel path: inductor_cache/bs/cbsdtmzxll46h3jupwjbs6ev2xtkf5fkzt3lnuab4dnqp2lawvpp.py
# Topologically Sorted Source Nodes: [batch_norm_1, relu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_1
#   relu_1 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
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
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 48)
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


# kernel path: inductor_cache/xt/cxtul34xsjxixpoxj5keqhdvupwolrb5bbhvgw2tf64ntalxlzsj.py
# Topologically Sorted Source Nodes: [concated_features_1, batch_norm_2, relu_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_5, mul_7, mul_8, sub_2
#   concated_features_1 => cat
#   relu_2 => relu_2
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 36)
    x0 = (xindex % 4096)
    x2 = xindex // 147456
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-24) + x1) + 49152*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/vc/cvct7noqvxnuyytxw5q6qidfyd7cyislxi5xbf34yyo3juo7ud5u.py
# Topologically Sorted Source Nodes: [concated_features_2, batch_norm_4, relu_4], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_4 => add_9, mul_13, mul_14, sub_4
#   concated_features_2 => cat_1
#   relu_4 => relu_4
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 48)
    x0 = (xindex % 4096)
    x2 = xindex // 196608
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-24) + x1) + 49152*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 4096*((-36) + x1) + 49152*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/7g/c7g4yuonxlnkx4xduszw5mhzb2wbxndbaiym43lidtnzilvgckaf.py
# Topologically Sorted Source Nodes: [concated_features_3, batch_norm_6, relu_6], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_6 => add_13, mul_19, mul_20, sub_6
#   concated_features_3 => cat_2
#   relu_6 => relu_6
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_2, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 60)
    x0 = (xindex % 4096)
    x2 = xindex // 245760
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-24) + x1) + 49152*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4096*((-36) + x1) + 49152*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 60, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 4096*((-48) + x1) + 49152*x2), tmp16, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/33/c33splvc6jihmtkroqjhsalbqrouvgf7unrpkt2gt3ergv6uo2rx.py
# Topologically Sorted Source Nodes: [concated_features_4, batch_norm_8, relu_8], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_8 => add_17, mul_25, mul_26, sub_8
#   concated_features_4 => cat_3
#   relu_8 => relu_8
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8], 1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_3, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 72)
    x0 = (xindex % 4096)
    x2 = xindex // 294912
    x3 = xindex
    tmp29 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-24) + x1) + 49152*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4096*((-36) + x1) + 49152*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 60, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4096*((-48) + x1) + 49152*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 72, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 4096*((-60) + x1) + 49152*x2), tmp21, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/mz/cmzxtqqwjlj6okgkjfwqaztzhbzy3zqzdnnozwhzjbafmxc5y2xm.py
# Topologically Sorted Source Nodes: [concated_features_5, batch_norm_10, relu_10], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_10 => add_21, mul_31, mul_32, sub_10
#   concated_features_5 => cat_4
#   relu_10 => relu_10
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10], 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1376256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 84)
    x0 = (xindex % 4096)
    x2 = xindex // 344064
    x3 = xindex
    tmp35 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-24) + x1) + 49152*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4096*((-36) + x1) + 49152*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 60, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4096*((-48) + x1) + 49152*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 72, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4096*((-60) + x1) + 49152*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 84, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 4096*((-72) + x1) + 49152*x2), tmp26, other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tl.store(out_ptr0 + (x3), tmp34, None)
    tl.store(out_ptr1 + (x3), tmp51, None)
''', device_str='cuda')


# kernel path: inductor_cache/5o/c5orwq3xszapbrdcnggzhttspmg22uqxndng3nm44jsmtehmgkjx.py
# Topologically Sorted Source Nodes: [concated_features_6, batch_norm_12, relu_12], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_12 => add_25, mul_37, mul_38, sub_12
#   concated_features_6 => cat_5
#   relu_12 => relu_12
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 96)
    x0 = (xindex % 4096)
    x2 = xindex // 393216
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-24) + x1) + 49152*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4096*((-36) + x1) + 49152*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 60, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4096*((-48) + x1) + 49152*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 72, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4096*((-60) + x1) + 49152*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 84, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 4096*((-72) + x1) + 49152*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 96, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 4096*((-84) + x1) + 49152*x2), tmp31, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, None)
    tl.store(out_ptr1 + (x3), tmp57, None)
''', device_str='cuda')


# kernel path: inductor_cache/kv/ckvrg6ukdlz7ycgfztzc4gdny7k6tbobw3jc2vbsqbf3y4it3fas.py
# Topologically Sorted Source Nodes: [concated_features_7, batch_norm_14, relu_14], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_14 => add_29, mul_43, mul_44, sub_14
#   concated_features_7 => cat_6
#   relu_14 => relu_14
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14], 1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_6, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 108)
    x0 = (xindex % 4096)
    x2 = xindex // 442368
    x3 = xindex
    tmp47 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 98304*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-24) + x1) + 49152*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4096*((-36) + x1) + 49152*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 60, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4096*((-48) + x1) + 49152*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 72, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4096*((-60) + x1) + 49152*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 84, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 4096*((-72) + x1) + 49152*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 96, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr6 + (x0 + 4096*((-84) + x1) + 49152*x2), tmp34, other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 108, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr7 + (x0 + 4096*((-96) + x1) + 49152*x2), tmp36, other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp14, tmp15, tmp43)
    tmp45 = tl.where(tmp9, tmp10, tmp44)
    tmp46 = tl.where(tmp4, tmp5, tmp45)
    tmp48 = tmp46 - tmp47
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tl.full([1], 1, tl.int32)
    tmp54 = tmp53 / tmp52
    tmp55 = 1.0
    tmp56 = tmp54 * tmp55
    tmp57 = tmp48 * tmp56
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full([1], 0, tl.int32)
    tmp63 = triton_helpers.maximum(tmp62, tmp61)
    tl.store(out_ptr0 + (x3), tmp46, None)
    tl.store(out_ptr1 + (x3), tmp63, None)
''', device_str='cuda')


# kernel path: inductor_cache/rq/crq47sonzxvr2znqkiaas5bv2wsuvkzpfwml2nv2yftkccr76ppr.py
# Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_10 => cat_9
#   concated_features_11 => cat_10
#   concated_features_12 => cat_11
#   concated_features_8 => cat_7
#   concated_features_9 => cat_8
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16], 1), kwargs = {})
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18], 1), kwargs = {})
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20], 1), kwargs = {})
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22], 1), kwargs = {})
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24], 1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 98304)
    x1 = xindex // 98304
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 491520*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 540672*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 589824*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 638976*x1), tmp0, None)
    tl.store(out_ptr4 + (x0 + 688128*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/a5/ca5wgymnh4x2rmkgt74zjlgrsrrw5gos6h2pyet6amuteuo6jja7.py
# Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_10 => cat_9
#   concated_features_11 => cat_10
#   concated_features_12 => cat_11
#   concated_features_8 => cat_7
#   concated_features_9 => cat_8
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16], 1), kwargs = {})
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18], 1), kwargs = {})
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20], 1), kwargs = {})
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22], 1), kwargs = {})
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24], 1), kwargs = {})
triton_poi_fused_cat_10 = async_compile.triton('triton_poi_fused_cat_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_10(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 491520*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 540672*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 589824*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 638976*x1), tmp0, None)
    tl.store(out_ptr4 + (x0 + 688128*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/xm/cxmyisbkfvm6umsbsc7a4joezctmrui4q6iwcmmlo53h6fs4l3uv.py
# Topologically Sorted Source Nodes: [batch_norm_16, relu_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_16 => add_33, mul_49, mul_50, sub_16
#   relu_16 => relu_16
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %unsqueeze_129), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 120)
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


# kernel path: inductor_cache/bc/cbcxa3viipvylh5ujdeplx7s2xnnjno2zksprxd2ahhd6m3msbxd.py
# Topologically Sorted Source Nodes: [concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_10 => cat_9
#   concated_features_11 => cat_10
#   concated_features_12 => cat_11
#   concated_features_9 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18], 1), kwargs = {})
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20], 1), kwargs = {})
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22], 1), kwargs = {})
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 540672*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 589824*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 638976*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 688128*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/lk/clkvwl7vquggyocw2toxjdxahhvug63vahs2g346pmwos5ebt4tf.py
# Topologically Sorted Source Nodes: [batch_norm_18, relu_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_18 => add_37, mul_55, mul_56, sub_18
#   relu_18 => relu_18
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_8, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2162688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 132)
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


# kernel path: inductor_cache/77/c77euvlg3tvazr2yi2baifsf34crpm5um2jhirc2cknkx6lzkdvv.py
# Topologically Sorted Source Nodes: [concated_features_10, concated_features_11, concated_features_12, concated_features_13], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_10 => cat_9
#   concated_features_11 => cat_10
#   concated_features_12 => cat_11
#   concated_features_13 => cat_12
# Graph fragment:
#   %cat_9 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20], 1), kwargs = {})
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22], 1), kwargs = {})
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24], 1), kwargs = {})
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 589824*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 638976*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 688128*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 737280*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/p4/cp4guo5fz4bxzqzrydbnpowqhcp3or4nqsu75dl57rymbkrluu3n.py
# Topologically Sorted Source Nodes: [batch_norm_20, relu_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_20 => add_41, mul_61, mul_62, sub_20
#   relu_20 => relu_20
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_9, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %relu_20 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 144)
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


# kernel path: inductor_cache/72/c726jyl3bnlsdc3b3deitvhljaodk2llvctvgs2ov5nr2vrueyw6.py
# Topologically Sorted Source Nodes: [concated_features_11, concated_features_12, concated_features_13, concated_features_14], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_11 => cat_10
#   concated_features_12 => cat_11
#   concated_features_13 => cat_12
#   concated_features_14 => cat_13
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22], 1), kwargs = {})
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24], 1), kwargs = {})
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26], 1), kwargs = {})
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 638976*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 688128*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 737280*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 786432*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/4t/c4ttn4nl6iykxyl5rxssknlbwi2gvqtb3untl35rwkr44anurq3x.py
# Topologically Sorted Source Nodes: [batch_norm_22, relu_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_22 => add_45, mul_67, mul_68, sub_22
#   relu_22 => relu_22
# Graph fragment:
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_10, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_45,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2555904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 156)
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


# kernel path: inductor_cache/wq/cwqeynbvpddjuk2igdtkpjhpu2izmlhlvwaasc6t3ocl3teexxd5.py
# Topologically Sorted Source Nodes: [concated_features_12, concated_features_13, concated_features_14, concated_features_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_12 => cat_11
#   concated_features_13 => cat_12
#   concated_features_14 => cat_13
#   concated_features_15 => cat_14
# Graph fragment:
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24], 1), kwargs = {})
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26], 1), kwargs = {})
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28], 1), kwargs = {})
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30], 1), kwargs = {})
triton_poi_fused_cat_18 = async_compile.triton('triton_poi_fused_cat_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 688128*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 737280*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 786432*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 835584*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/ky/ckyvb6jvtu6jozltte3il4axmxao3cd3haor3y4qisl3do7nwhyi.py
# Topologically Sorted Source Nodes: [batch_norm_24, relu_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_24 => add_49, mul_73, mul_74, sub_24
#   relu_24 => relu_24
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_11, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_49,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2752512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 168)
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


# kernel path: inductor_cache/3f/c3fjst7lb22zprxsdkrqfjoyz45alxsotngyb5b4iik7eewqoyo7.py
# Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_13 => cat_12
#   concated_features_14 => cat_13
#   concated_features_15 => cat_14
#   input_2 => cat_15
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26], 1), kwargs = {})
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28], 1), kwargs = {})
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30], 1), kwargs = {})
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30, %convolution_32], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 98304)
    x1 = xindex // 98304
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 737280*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 786432*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 835584*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 884736*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/wy/cwyqwiu3prqxv5aaoj2l4336ptlziwnebcgwmt5rd5f3u3f7x2lx.py
# Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_13 => cat_12
#   concated_features_14 => cat_13
#   concated_features_15 => cat_14
#   input_2 => cat_15
# Graph fragment:
#   %cat_12 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26], 1), kwargs = {})
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28], 1), kwargs = {})
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30], 1), kwargs = {})
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30, %convolution_32], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 737280*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 786432*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 835584*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 884736*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/th/cth2zu46es4w26kkctjfmeqsbpgwwtt2sxcz7ic6jdhkgn3pb4nz.py
# Topologically Sorted Source Nodes: [batch_norm_26, relu_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_26 => add_53, mul_79, mul_80, sub_26
#   relu_26 => relu_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_12, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2949120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 180)
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


# kernel path: inductor_cache/bl/cblhf7z3jtmmlcnzgrkbwbps3t76s4liiyw5hvhfv5mrivi5b5ep.py
# Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_14 => cat_13
#   concated_features_15 => cat_14
#   input_2 => cat_15
# Graph fragment:
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28], 1), kwargs = {})
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30], 1), kwargs = {})
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30, %convolution_32], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 786432*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 835584*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 884736*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/vo/cvopvzo27pmd62xh7xnnxtlcftg3c6n47qadtgxc4x6muj3gl34t.py
# Topologically Sorted Source Nodes: [batch_norm_28, relu_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_28 => add_57, mul_85, mul_86, sub_28
#   relu_28 => relu_28
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_13, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %relu_28 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 192)
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


# kernel path: inductor_cache/6h/c6hgbl5iffaom4vuewnv7w74qs5qly2ndri7cbse4w4t3oj3e3nk.py
# Topologically Sorted Source Nodes: [concated_features_15, input_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_15 => cat_14
#   input_2 => cat_15
# Graph fragment:
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30], 1), kwargs = {})
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30, %convolution_32], 1), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 835584*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 884736*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/sv/csvlf7vn6zfm6eb5ecdcjfwdwlwbzishpgyv3arpytpqow5difbs.py
# Topologically Sorted Source Nodes: [batch_norm_30, relu_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_30 => add_61, mul_91, mul_92, sub_30
#   relu_30 => relu_30
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_14, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_61,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3342336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 204)
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


# kernel path: inductor_cache/jx/cjxqz6jscsmsklhrogf43ly3yc3yeqbzy42pdtjby3wljno5aqcd.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_2 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_4, %convolution_6, %convolution_8, %convolution_10, %convolution_12, %convolution_14, %convolution_16, %convolution_18, %convolution_20, %convolution_22, %convolution_24, %convolution_26, %convolution_28, %convolution_30, %convolution_32], 1), kwargs = {})
triton_poi_fused_cat_27 = async_compile.triton('triton_poi_fused_cat_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 49152)
    x1 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 884736*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/h5/ch5iphzp3f6kuhwz3z46o4teglb7g3sgkbrwls2ez33jjntnmwhn.py
# Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_3 => add_65, mul_97, mul_98, sub_32
#   input_4 => relu_32
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_15, %unsqueeze_257), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_261), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_263), kwargs = {})
#   %relu_32 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 216)
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


# kernel path: inductor_cache/as/caswtr6w63e7bhpfwjfqon3eo6p577iamam7nc3jzishzazsho3s.py
# Topologically Sorted Source Nodes: [input_6, batch_norm_33, relu_33, concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   batch_norm_33 => add_67, mul_100, mul_101, sub_33
#   concated_features_24 => cat_23
#   concated_features_25 => cat_24
#   concated_features_26 => cat_25
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
#   input_6 => avg_pool2d
#   input_7 => cat_31
#   relu_33 => relu_33
# Graph fragment:
#   %avg_pool2d : [num_users=18] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_33, [2, 2], [2, 2]), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d, %unsqueeze_265), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %relu_33 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_67,), kwargs = {})
#   %cat_23 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49], 1), kwargs = {})
#   %cat_24 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51], 1), kwargs = {})
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53], 1), kwargs = {})
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63, %convolution_65], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x6 = xindex
    x3 = ((xindex // 1024) % 108)
    x4 = xindex // 110592
    x5 = (xindex % 110592)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x6), tmp8, None)
    tl.store(out_ptr1 + (x6), tmp25, None)
    tl.store(out_ptr2 + (x5 + 208896*x4), tmp8, None)
    tl.store(out_ptr3 + (x5 + 221184*x4), tmp8, None)
    tl.store(out_ptr4 + (x5 + 233472*x4), tmp8, None)
    tl.store(out_ptr5 + (x5 + 245760*x4), tmp8, None)
    tl.store(out_ptr6 + (x5 + 258048*x4), tmp8, None)
    tl.store(out_ptr7 + (x5 + 270336*x4), tmp8, None)
    tl.store(out_ptr8 + (x5 + 282624*x4), tmp8, None)
    tl.store(out_ptr9 + (x5 + 294912*x4), tmp8, None)
    tl.store(out_ptr10 + (x5 + 307200*x4), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/nl/cnlknq37u76xwzrxjgziej55sxbmi4eomojqj7n4dut5fui6zcyo.py
# Topologically Sorted Source Nodes: [batch_norm_34, relu_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_34 => add_69, mul_103, mul_104, sub_34
#   relu_34 => relu_34
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_273), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %relu_34 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_69,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 48)
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


# kernel path: inductor_cache/sn/csnh2mtilfayw3q2j6pyysshlqde2c6wxwyp4esnzctaepdvegeb.py
# Topologically Sorted Source Nodes: [concated_features_17, batch_norm_35, relu_35], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_35 => add_71, mul_106, mul_107, sub_35
#   concated_features_17 => cat_16
#   relu_35 => relu_35
# Graph fragment:
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35], 1), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_16, %unsqueeze_281), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %unsqueeze_285), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %unsqueeze_287), kwargs = {})
#   %relu_35 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_71,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 120)
    x0 = (xindex % 1024)
    x2 = xindex // 122880
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 110592*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 1024*((-108) + x1) + 12288*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/x6/cx6t7zndb7qkgsu34pmngsx2oovxmxu5smo2yqfp7mgvob4gi5ap.py
# Topologically Sorted Source Nodes: [concated_features_18, batch_norm_37, relu_37], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_37 => add_75, mul_112, mul_113, sub_37
#   concated_features_18 => cat_17
#   relu_37 => relu_37
# Graph fragment:
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37], 1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_17, %unsqueeze_297), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %relu_37 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_75,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 540672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 132)
    x0 = (xindex % 1024)
    x2 = xindex // 135168
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 110592*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-108) + x1) + 12288*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 132, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 1024*((-120) + x1) + 12288*x2), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/fl/cflmqpfjhuwqwbkmxy347bn4oc7eatx72vil5cgvu2dbfokpw73c.py
# Topologically Sorted Source Nodes: [concated_features_19, batch_norm_39, relu_39], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_39 => add_79, mul_118, mul_119, sub_39
#   concated_features_19 => cat_18
#   relu_39 => relu_39
# Graph fragment:
#   %cat_18 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39], 1), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_18, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_39 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_79,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 144)
    x0 = (xindex % 1024)
    x2 = xindex // 147456
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 110592*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-108) + x1) + 12288*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 132, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 1024*((-120) + x1) + 12288*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 144, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 1024*((-132) + x1) + 12288*x2), tmp16, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x3), tmp22, None)
    tl.store(out_ptr1 + (x3), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/f4/cf4aim2z7lxphiwjgxhvaarsjhbkl4wf7tm46oa7tnczywgkw2m7.py
# Topologically Sorted Source Nodes: [concated_features_20, batch_norm_41, relu_41], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_41 => add_83, mul_124, mul_125, sub_41
#   concated_features_20 => cat_19
#   relu_41 => relu_41
# Graph fragment:
#   %cat_19 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41], 1), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_19, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_83,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 638976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 156)
    x0 = (xindex % 1024)
    x2 = xindex // 159744
    x3 = xindex
    tmp29 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 110592*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-108) + x1) + 12288*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 132, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 1024*((-120) + x1) + 12288*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 144, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 1024*((-132) + x1) + 12288*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 156, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 1024*((-144) + x1) + 12288*x2), tmp21, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp45, None)
''', device_str='cuda')


# kernel path: inductor_cache/62/c62v3r2553fuexjo6fwqwmsg7hsa5qt5syo62msr3snlqk2xnhxx.py
# Topologically Sorted Source Nodes: [concated_features_21, batch_norm_43, relu_43], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_43 => add_87, mul_130, mul_131, sub_43
#   concated_features_21 => cat_20
#   relu_43 => relu_43
# Graph fragment:
#   %cat_20 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43], 1), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_20, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_43 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 688128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 168)
    x0 = (xindex % 1024)
    x2 = xindex // 172032
    x3 = xindex
    tmp35 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 110592*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-108) + x1) + 12288*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 132, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 1024*((-120) + x1) + 12288*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 144, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 1024*((-132) + x1) + 12288*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 156, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 1024*((-144) + x1) + 12288*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 168, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 1024*((-156) + x1) + 12288*x2), tmp26, other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tl.store(out_ptr0 + (x3), tmp34, None)
    tl.store(out_ptr1 + (x3), tmp51, None)
''', device_str='cuda')


# kernel path: inductor_cache/74/c747mtuteayi3wmn2mr25ufwwjrvindfbkhybck3gd5ri3ehii7w.py
# Topologically Sorted Source Nodes: [concated_features_22, batch_norm_45, relu_45], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_45 => add_91, mul_136, mul_137, sub_45
#   concated_features_22 => cat_21
#   relu_45 => relu_45
# Graph fragment:
#   %cat_21 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45], 1), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_21, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 737280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 180)
    x0 = (xindex % 1024)
    x2 = xindex // 184320
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 110592*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-108) + x1) + 12288*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 132, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 1024*((-120) + x1) + 12288*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 144, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 1024*((-132) + x1) + 12288*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 156, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 1024*((-144) + x1) + 12288*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 168, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 1024*((-156) + x1) + 12288*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 180, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 1024*((-168) + x1) + 12288*x2), tmp31, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, None)
    tl.store(out_ptr1 + (x3), tmp57, None)
''', device_str='cuda')


# kernel path: inductor_cache/pv/cpvn7voi4wiz364ojmdolumhqawjz4li242pyocduf3ty5ccltyo.py
# Topologically Sorted Source Nodes: [concated_features_23, batch_norm_47, relu_47], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_47 => add_95, mul_142, mul_143, sub_47
#   concated_features_23 => cat_22
#   relu_47 => relu_47
# Graph fragment:
#   %cat_22 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47], 1), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_22, %unsqueeze_377), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_379), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, %unsqueeze_381), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, %unsqueeze_383), kwargs = {})
#   %relu_47 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_95,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 192)
    x0 = (xindex % 1024)
    x2 = xindex // 196608
    x3 = xindex
    tmp47 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 110592*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 1024*((-108) + x1) + 12288*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 132, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 1024*((-120) + x1) + 12288*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 144, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 1024*((-132) + x1) + 12288*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 156, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 1024*((-144) + x1) + 12288*x2), tmp24, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 168, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 1024*((-156) + x1) + 12288*x2), tmp29, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 180, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr6 + (x0 + 1024*((-168) + x1) + 12288*x2), tmp34, other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 192, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr7 + (x0 + 1024*((-180) + x1) + 12288*x2), tmp36, other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp14, tmp15, tmp43)
    tmp45 = tl.where(tmp9, tmp10, tmp44)
    tmp46 = tl.where(tmp4, tmp5, tmp45)
    tmp48 = tmp46 - tmp47
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tl.full([1], 1, tl.int32)
    tmp54 = tmp53 / tmp52
    tmp55 = 1.0
    tmp56 = tmp54 * tmp55
    tmp57 = tmp48 * tmp56
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full([1], 0, tl.int32)
    tmp63 = triton_helpers.maximum(tmp62, tmp61)
    tl.store(out_ptr0 + (x3), tmp46, None)
    tl.store(out_ptr1 + (x3), tmp63, None)
''', device_str='cuda')


# kernel path: inductor_cache/xa/cxa26yo5ez6dp44q34dndcnx3pd3x6spxukaq5prddtwhnkv5454.py
# Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_24 => cat_23
#   concated_features_25 => cat_24
#   concated_features_26 => cat_25
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
# Graph fragment:
#   %cat_23 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49], 1), kwargs = {})
#   %cat_24 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51], 1), kwargs = {})
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53], 1), kwargs = {})
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57], 1), kwargs = {})
triton_poi_fused_cat_38 = async_compile.triton('triton_poi_fused_cat_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_38(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 208896*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 221184*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 233472*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 245760*x1), tmp0, None)
    tl.store(out_ptr4 + (x0 + 258048*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/k2/ck2gzww5juvm57zrvomakquc55sqsiuviddcmmljldzb3bgyusj2.py
# Topologically Sorted Source Nodes: [batch_norm_49, relu_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_49 => add_99, mul_148, mul_149, sub_49
#   relu_49 => relu_49
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_23, %unsqueeze_393), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_397), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_399), kwargs = {})
#   %relu_49 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_99,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 835584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 204)
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


# kernel path: inductor_cache/e3/ce3us5fd34ceor4h4u5qtpjjwqx2qfulvylcivhxvbermax5g2u7.py
# Topologically Sorted Source Nodes: [concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_25 => cat_24
#   concated_features_26 => cat_25
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
# Graph fragment:
#   %cat_24 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51], 1), kwargs = {})
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53], 1), kwargs = {})
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 221184*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 233472*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 245760*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 258048*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/6s/c6sb7ou26bnu6gl5rfq6cv3nooivwhhlrtkfyv7gfryvh24kes2p.py
# Topologically Sorted Source Nodes: [batch_norm_51, relu_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_51 => add_103, mul_154, mul_155, sub_51
#   relu_51 => relu_51
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_24, %unsqueeze_409), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %unsqueeze_413), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %unsqueeze_415), kwargs = {})
#   %relu_51 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_103,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 216)
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


# kernel path: inductor_cache/cb/ccbqqedn6ovpnpilpbk3pnh3j36pywpbiomxr3oaagrkglvwuuzv.py
# Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_26 => cat_25
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
# Graph fragment:
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53], 1), kwargs = {})
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59], 1), kwargs = {})
triton_poi_fused_cat_42 = async_compile.triton('triton_poi_fused_cat_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_42(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 233472*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 245760*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 258048*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 270336*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg664ix2wauanejvgw72m6cssgbzcs3tntz242dil7sh4mbco3q3.py
# Topologically Sorted Source Nodes: [batch_norm_53, relu_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_53 => add_107, mul_160, mul_161, sub_53
#   relu_53 => relu_53
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_25, %unsqueeze_425), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_429), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_431), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 933888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 228)
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


# kernel path: inductor_cache/bj/cbjfpxtfndxwk4dbhcyezqogijgxu5aqjwiozcmpxv24s2xtlame.py
# Topologically Sorted Source Nodes: [concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_27 => cat_26
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
# Graph fragment:
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55], 1), kwargs = {})
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61], 1), kwargs = {})
triton_poi_fused_cat_44 = async_compile.triton('triton_poi_fused_cat_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_44(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 245760*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 258048*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 270336*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 282624*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/rd/crddh4vb4kwrvgtj7hj5rmthwk3dsawbv6yq63su7q7hwn5sqins.py
# Topologically Sorted Source Nodes: [batch_norm_55, relu_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_55 => add_111, mul_166, mul_167, sub_55
#   relu_55 => relu_55
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_26, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 240)
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


# kernel path: inductor_cache/64/c64adtmfzfjeucdknxas7a2g3brg375ksm2m73fdg7wtwl6fh6cd.py
# Topologically Sorted Source Nodes: [concated_features_28, concated_features_29, concated_features_30, concated_features_31], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_28 => cat_27
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
# Graph fragment:
#   %cat_27 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57], 1), kwargs = {})
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63], 1), kwargs = {})
triton_poi_fused_cat_46 = async_compile.triton('triton_poi_fused_cat_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_46(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 258048*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 270336*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 282624*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 294912*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/rp/crpfr7kuqesjjuqrndhgra3koqx54xnq2aumtxrbxwq2iqulaon6.py
# Topologically Sorted Source Nodes: [batch_norm_57, relu_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_57 => add_115, mul_172, mul_173, sub_57
#   relu_57 => relu_57
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_27, %unsqueeze_457), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_461), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_463), kwargs = {})
#   %relu_57 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_115,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1032192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 252)
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


# kernel path: inductor_cache/rh/crhlncdigpb4vrduzr3hix4ln5e2bwkxh2q4xvhnxh5olf74cfbs.py
# Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_29 => cat_28
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
#   input_7 => cat_31
# Graph fragment:
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59], 1), kwargs = {})
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63, %convolution_65], 1), kwargs = {})
triton_poi_fused_cat_48 = async_compile.triton('triton_poi_fused_cat_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_48(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 270336*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 282624*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 294912*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 307200*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/2g/c2g2f3go5pfd7rc5clxptonecesm6rxydrb2psapccrnffc2ibc2.py
# Topologically Sorted Source Nodes: [batch_norm_59, relu_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_59 => add_119, mul_178, mul_179, sub_59
#   relu_59 => relu_59
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_28, %unsqueeze_473), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_477), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_479), kwargs = {})
#   %relu_59 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_119,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1081344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 264)
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


# kernel path: inductor_cache/h5/ch5333gxk2edsb4oh3i5bocosujdywjd6w5o7w7aqto6q27m35nl.py
# Topologically Sorted Source Nodes: [concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_30 => cat_29
#   concated_features_31 => cat_30
#   input_7 => cat_31
# Graph fragment:
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61], 1), kwargs = {})
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63, %convolution_65], 1), kwargs = {})
triton_poi_fused_cat_50 = async_compile.triton('triton_poi_fused_cat_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_50(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 282624*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 294912*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 307200*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/fz/cfz4ti6ntrd7om53nahwt5pv5jri4ljt2whf3thkfdhv6glkpnua.py
# Topologically Sorted Source Nodes: [batch_norm_61, relu_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_61 => add_123, mul_184, mul_185, sub_61
#   relu_61 => relu_61
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_29, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_123,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1130496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 276)
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


# kernel path: inductor_cache/rd/crdl7bnauhuiilke7cxuttwbo5kfgam4drnqw6zmznttmg7tqkrx.py
# Topologically Sorted Source Nodes: [concated_features_31, input_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_31 => cat_30
#   input_7 => cat_31
# Graph fragment:
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63], 1), kwargs = {})
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63, %convolution_65], 1), kwargs = {})
triton_poi_fused_cat_52 = async_compile.triton('triton_poi_fused_cat_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_52(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 294912*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 307200*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/b2/cb2seriedkegcun7taxamfj7ifzrdecaner3km3ccemyabcduwby.py
# Topologically Sorted Source Nodes: [batch_norm_63, relu_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_63 => add_127, mul_190, mul_191, sub_63
#   relu_63 => relu_63
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_30, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %relu_63 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_127,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 288)
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


# kernel path: inductor_cache/j2/cj2vglknr6vhim45wuldqlwr43r6bbygvf7mg4klhsaabb7sgkoa.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_7 => cat_31
# Graph fragment:
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_35, %convolution_37, %convolution_39, %convolution_41, %convolution_43, %convolution_45, %convolution_47, %convolution_49, %convolution_51, %convolution_53, %convolution_55, %convolution_57, %convolution_59, %convolution_61, %convolution_63, %convolution_65], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 307200*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/5d/c5dx36myk44fy2pe6hls5ud3anucossx2d5udhdq3u5v6yrrll73.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_8 => add_131, mul_196, mul_197, sub_65
#   input_9 => relu_65
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_31, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %relu_65 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_131,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1228800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 300)
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


# kernel path: inductor_cache/vw/cvwrv6263lyasnwuxauqgchk6beiw3szhqsj6sktqaaaj4qcy255.py
# Topologically Sorted Source Nodes: [input_11, batch_norm_66, relu_66, concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44, concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   batch_norm_66 => add_133, mul_199, mul_200, sub_66
#   concated_features_40 => cat_39
#   concated_features_41 => cat_40
#   concated_features_42 => cat_41
#   concated_features_43 => cat_42
#   concated_features_44 => cat_43
#   concated_features_45 => cat_44
#   concated_features_46 => cat_45
#   concated_features_47 => cat_46
#   input_11 => avg_pool2d_1
#   input_12 => cat_47
#   relu_66 => relu_66
# Graph fragment:
#   %avg_pool2d_1 : [num_users=18] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_66, [2, 2], [2, 2]), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_1, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %relu_66 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_133,), kwargs = {})
#   %cat_39 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82], 1), kwargs = {})
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84], 1), kwargs = {})
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
#   %cat_42 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88], 1), kwargs = {})
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90], 1), kwargs = {})
#   %cat_44 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92], 1), kwargs = {})
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94], 1), kwargs = {})
#   %cat_46 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96], 1), kwargs = {})
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96, %convolution_98], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_56', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 153600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x6 = xindex
    x3 = ((xindex // 256) % 150)
    x4 = xindex // 38400
    x5 = (xindex % 38400)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x6), tmp8, xmask)
    tl.store(out_ptr1 + (x6), tmp25, xmask)
    tl.store(out_ptr2 + (x5 + 62976*x4), tmp8, xmask)
    tl.store(out_ptr3 + (x5 + 66048*x4), tmp8, xmask)
    tl.store(out_ptr4 + (x5 + 69120*x4), tmp8, xmask)
    tl.store(out_ptr5 + (x5 + 72192*x4), tmp8, xmask)
    tl.store(out_ptr6 + (x5 + 75264*x4), tmp8, xmask)
    tl.store(out_ptr7 + (x5 + 78336*x4), tmp8, xmask)
    tl.store(out_ptr8 + (x5 + 81408*x4), tmp8, xmask)
    tl.store(out_ptr9 + (x5 + 84480*x4), tmp8, xmask)
    tl.store(out_ptr10 + (x5 + 87552*x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zp/czpu7s4s66fln3ket4rv54madsynjb5cr7tw2eewbe43pmyenx6k.py
# Topologically Sorted Source Nodes: [batch_norm_67, relu_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_67 => add_135, mul_202, mul_203, sub_67
#   relu_67 => relu_67
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %relu_67 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 48)
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


# kernel path: inductor_cache/xq/cxqzckdvt6gwwyyp6ilolpe4pglekdjukjgjygxrtvgwxoo5p2zs.py
# Topologically Sorted Source Nodes: [concated_features_33, batch_norm_68, relu_68], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_68 => add_137, mul_205, mul_206, sub_68
#   concated_features_33 => cat_32
#   relu_68 => relu_68
# Graph fragment:
#   %cat_32 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68], 1), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_32, %unsqueeze_545), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_549), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_551), kwargs = {})
#   %relu_68 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_137,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 165888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 162)
    x0 = (xindex % 256)
    x2 = xindex // 41472
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 38400*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 162, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-150) + x1) + 3072*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pp/cppmrcb7e6qzdvoddet4ytegxik7q4pyew55nytixnasr2kolkit.py
# Topologically Sorted Source Nodes: [concated_features_34, batch_norm_70, relu_70], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_70 => add_141, mul_211, mul_212, sub_70
#   concated_features_34 => cat_33
#   relu_70 => relu_70
# Graph fragment:
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70], 1), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_33, %unsqueeze_561), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_211, %unsqueeze_565), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_212, %unsqueeze_567), kwargs = {})
#   %relu_70 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_141,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 178176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 174)
    x0 = (xindex % 256)
    x2 = xindex // 44544
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 38400*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 162, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-150) + x1) + 3072*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 174, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 256*((-162) + x1) + 3072*x2), tmp11 & xmask, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zb/czbetnc36ha3btudjhilvuwb2dycb6t75qymsmay7466xuth72je.py
# Topologically Sorted Source Nodes: [concated_features_35, batch_norm_72, relu_72], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_72 => add_145, mul_217, mul_218, sub_72
#   concated_features_35 => cat_34
#   relu_72 => relu_72
# Graph fragment:
#   %cat_34 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72], 1), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_34, %unsqueeze_577), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_581), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_583), kwargs = {})
#   %relu_72 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_145,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 190464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 186)
    x0 = (xindex % 256)
    x2 = xindex // 47616
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 38400*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 162, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-150) + x1) + 3072*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 174, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-162) + x1) + 3072*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 186, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 256*((-174) + x1) + 3072*x2), tmp16 & xmask, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5r/c5r7mqeyob2fbmcbqtytch6ogyzc2o2d6kq4qgpz5ydmi5l26xzf.py
# Topologically Sorted Source Nodes: [concated_features_36, batch_norm_74, relu_74], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_74 => add_149, mul_223, mul_224, sub_74
#   concated_features_36 => cat_35
#   relu_74 => relu_74
# Graph fragment:
#   %cat_35 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74], 1), kwargs = {})
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_35, %unsqueeze_593), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_595), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_223, %unsqueeze_597), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_224, %unsqueeze_599), kwargs = {})
#   %relu_74 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_149,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 202752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 198)
    x0 = (xindex % 256)
    x2 = xindex // 50688
    x3 = xindex
    tmp29 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 38400*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 162, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-150) + x1) + 3072*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 174, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-162) + x1) + 3072*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 186, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 256*((-174) + x1) + 3072*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 198, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x0 + 256*((-186) + x1) + 3072*x2), tmp21 & xmask, other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kf/ckfue2fcavu4aug3h2hqqh5moetkfqhkf6abk6azfowccjaemkkb.py
# Topologically Sorted Source Nodes: [concated_features_37, batch_norm_76, relu_76], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_76 => add_153, mul_229, mul_230, sub_76
#   concated_features_37 => cat_36
#   relu_76 => relu_76
# Graph fragment:
#   %cat_36 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76], 1), kwargs = {})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_36, %unsqueeze_609), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %unsqueeze_613), kwargs = {})
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %unsqueeze_615), kwargs = {})
#   %relu_76 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_153,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 215040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 210)
    x0 = (xindex % 256)
    x2 = xindex // 53760
    x3 = xindex
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 38400*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 162, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-150) + x1) + 3072*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 174, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-162) + x1) + 3072*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 186, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 256*((-174) + x1) + 3072*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 198, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 256*((-186) + x1) + 3072*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 210, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 256*((-198) + x1) + 3072*x2), tmp26 & xmask, other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = triton_helpers.maximum(tmp50, tmp49)
    tl.store(out_ptr0 + (x3), tmp34, xmask)
    tl.store(out_ptr1 + (x3), tmp51, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/el/celpptthg6ez27wc3oycnreqokr5h7eoknnr2k6bh6odpromn3nq.py
# Topologically Sorted Source Nodes: [concated_features_38, batch_norm_78, relu_78], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_78 => add_157, mul_235, mul_236, sub_78
#   concated_features_38 => cat_37
#   relu_78 => relu_78
# Graph fragment:
#   %cat_37 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78], 1), kwargs = {})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_37, %unsqueeze_625), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_629), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_631), kwargs = {})
#   %relu_78 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 227328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 222)
    x0 = (xindex % 256)
    x2 = xindex // 56832
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 38400*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 162, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-150) + x1) + 3072*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 174, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-162) + x1) + 3072*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 186, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 256*((-174) + x1) + 3072*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 198, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 256*((-186) + x1) + 3072*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 210, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 256*((-198) + x1) + 3072*x2), tmp29 & xmask, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 222, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 256*((-210) + x1) + 3072*x2), tmp31 & xmask, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, xmask)
    tl.store(out_ptr1 + (x3), tmp57, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qh/cqhtc6fugipwr26wsuf5aos6dpxu5qooi6p5y2btdzqycmu6qhmt.py
# Topologically Sorted Source Nodes: [concated_features_39, batch_norm_80, relu_80], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_80 => add_161, mul_241, mul_242, sub_80
#   concated_features_39 => cat_38
#   relu_80 => relu_80
# Graph fragment:
#   %cat_38 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80], 1), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_38, %unsqueeze_641), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_645), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_647), kwargs = {})
#   %relu_80 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_161,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 239616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 234)
    x0 = (xindex % 256)
    x2 = xindex // 59904
    x3 = xindex
    tmp47 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 38400*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 162, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-150) + x1) + 3072*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 174, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 256*((-162) + x1) + 3072*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 186, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 256*((-174) + x1) + 3072*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 198, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 256*((-186) + x1) + 3072*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 210, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 256*((-198) + x1) + 3072*x2), tmp29 & xmask, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 222, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr6 + (x0 + 256*((-210) + x1) + 3072*x2), tmp34 & xmask, other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 234, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr7 + (x0 + 256*((-222) + x1) + 3072*x2), tmp36 & xmask, other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp14, tmp15, tmp43)
    tmp45 = tl.where(tmp9, tmp10, tmp44)
    tmp46 = tl.where(tmp4, tmp5, tmp45)
    tmp48 = tmp46 - tmp47
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tl.full([1], 1, tl.int32)
    tmp54 = tmp53 / tmp52
    tmp55 = 1.0
    tmp56 = tmp54 * tmp55
    tmp57 = tmp48 * tmp56
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full([1], 0, tl.int32)
    tmp63 = triton_helpers.maximum(tmp62, tmp61)
    tl.store(out_ptr0 + (x3), tmp46, xmask)
    tl.store(out_ptr1 + (x3), tmp63, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7s/c7sv5x5bfq3eh637lhb273lhclkswrhx7tbkmxsps7gen2fugp5n.py
# Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_40 => cat_39
#   concated_features_41 => cat_40
#   concated_features_42 => cat_41
#   concated_features_43 => cat_42
#   concated_features_44 => cat_43
# Graph fragment:
#   %cat_39 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82], 1), kwargs = {})
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84], 1), kwargs = {})
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
#   %cat_42 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88], 1), kwargs = {})
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90], 1), kwargs = {})
triton_poi_fused_cat_65 = async_compile.triton('triton_poi_fused_cat_65', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_65', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_65(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 62976*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 66048*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 69120*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 72192*x1), tmp0, None)
    tl.store(out_ptr4 + (x0 + 75264*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/nh/cnhjrsaksmobs73agwzgqtc4huy6n5wa74fe2rmunom22uninqh4.py
# Topologically Sorted Source Nodes: [batch_norm_82, relu_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_82 => add_165, mul_247, mul_248, sub_82
#   relu_82 => relu_82
# Graph fragment:
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_39, %unsqueeze_657), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_659), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_247, %unsqueeze_661), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_248, %unsqueeze_663), kwargs = {})
#   %relu_82 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_165,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_66 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_66', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 251904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 246)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k4/ck4f7eqreqslvg62ragxz6sgzincztonmgkbnkgjfeaqid4zvmgt.py
# Topologically Sorted Source Nodes: [concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_41 => cat_40
#   concated_features_42 => cat_41
#   concated_features_43 => cat_42
#   concated_features_44 => cat_43
# Graph fragment:
#   %cat_40 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84], 1), kwargs = {})
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
#   %cat_42 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88], 1), kwargs = {})
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90], 1), kwargs = {})
triton_poi_fused_cat_67 = async_compile.triton('triton_poi_fused_cat_67', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_67', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_67(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 66048*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 69120*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 72192*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 75264*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/kg/ckghujktm4agvlgjoml3qnlpo52h3qw77tu4zl2rm56yev3ehyg5.py
# Topologically Sorted Source Nodes: [batch_norm_84, relu_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_84 => add_169, mul_253, mul_254, sub_84
#   relu_84 => relu_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_40, %unsqueeze_673), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_253, %unsqueeze_677), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %unsqueeze_679), kwargs = {})
#   %relu_84 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_169,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_68 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_68', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_68', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 264192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 258)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ny/cnyeyspbws3f3n6eoakq27fga7rbe5r7ryh7smxhgyveycz5css2.py
# Topologically Sorted Source Nodes: [concated_features_42, concated_features_43, concated_features_44, concated_features_45], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_42 => cat_41
#   concated_features_43 => cat_42
#   concated_features_44 => cat_43
#   concated_features_45 => cat_44
# Graph fragment:
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86], 1), kwargs = {})
#   %cat_42 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88], 1), kwargs = {})
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90], 1), kwargs = {})
#   %cat_44 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92], 1), kwargs = {})
triton_poi_fused_cat_69 = async_compile.triton('triton_poi_fused_cat_69', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_69(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 69120*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 72192*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 75264*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 78336*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/mh/cmhijzgmlfkc67qtadou7wsewrobk6itk5fvzdvzwxqmt3mws64g.py
# Topologically Sorted Source Nodes: [batch_norm_86, relu_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_86 => add_173, mul_259, mul_260, sub_86
#   relu_86 => relu_86
# Graph fragment:
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_41, %unsqueeze_689), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_259, %unsqueeze_693), kwargs = {})
#   %add_173 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_260, %unsqueeze_695), kwargs = {})
#   %relu_86 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_173,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_70 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_70', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_70', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 276480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 270)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rh/crhwpt2navxrmskwd7bhedchler65bx55wqs4dxzyh7ffekrkerx.py
# Topologically Sorted Source Nodes: [concated_features_43, concated_features_44, concated_features_45, concated_features_46], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_43 => cat_42
#   concated_features_44 => cat_43
#   concated_features_45 => cat_44
#   concated_features_46 => cat_45
# Graph fragment:
#   %cat_42 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88], 1), kwargs = {})
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90], 1), kwargs = {})
#   %cat_44 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92], 1), kwargs = {})
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94], 1), kwargs = {})
triton_poi_fused_cat_71 = async_compile.triton('triton_poi_fused_cat_71', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_71', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_71(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 72192*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 75264*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 78336*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 81408*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/7x/c7xhvu65yq7ollbaplhd7iu7repjugw5ejplnsmxhxvjbz6ag43a.py
# Topologically Sorted Source Nodes: [batch_norm_88, relu_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_88 => add_177, mul_265, mul_266, sub_88
#   relu_88 => relu_88
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_42, %unsqueeze_705), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_709), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_711), kwargs = {})
#   %relu_88 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_177,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_72 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_72', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_72', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_72(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 282)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gm/cgm2rulo47gbd4v6nm7klanrcmla5gj4ekqyubnekhnll2a2qtyw.py
# Topologically Sorted Source Nodes: [concated_features_44, concated_features_45, concated_features_46, concated_features_47], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_44 => cat_43
#   concated_features_45 => cat_44
#   concated_features_46 => cat_45
#   concated_features_47 => cat_46
# Graph fragment:
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90], 1), kwargs = {})
#   %cat_44 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92], 1), kwargs = {})
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94], 1), kwargs = {})
#   %cat_46 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96], 1), kwargs = {})
triton_poi_fused_cat_73 = async_compile.triton('triton_poi_fused_cat_73', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_73', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_73(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 75264*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 78336*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 81408*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 84480*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/ch/cch7xprd5y5siwlu7scgbqj7k55rvipsvcak3vbmfzqy7273uexb.py
# Topologically Sorted Source Nodes: [batch_norm_90, relu_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_90 => add_181, mul_271, mul_272, sub_90
#   relu_90 => relu_90
# Graph fragment:
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_43, %unsqueeze_721), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_725), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_727), kwargs = {})
#   %relu_90 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_181,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_74 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_74', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_74', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_74(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 294)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wq/cwquknnvchvryw4xkewz3f2fbrsstxeylzfz6jdj3p22xzbloy4q.py
# Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_45 => cat_44
#   concated_features_46 => cat_45
#   concated_features_47 => cat_46
#   input_12 => cat_47
# Graph fragment:
#   %cat_44 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92], 1), kwargs = {})
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94], 1), kwargs = {})
#   %cat_46 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96], 1), kwargs = {})
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96, %convolution_98], 1), kwargs = {})
triton_poi_fused_cat_75 = async_compile.triton('triton_poi_fused_cat_75', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_75', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_75(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 78336*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 81408*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 84480*x1), tmp0, None)
    tl.store(out_ptr3 + (x0 + 87552*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/rw/crwep3labihj3ik7mbizikmpitfrq2ef4nqwv7aaidw53wq7a7gc.py
# Topologically Sorted Source Nodes: [batch_norm_92, relu_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_92 => add_185, mul_277, mul_278, sub_92
#   relu_92 => relu_92
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_44, %unsqueeze_737), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %unsqueeze_741), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %unsqueeze_743), kwargs = {})
#   %relu_92 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_185,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_76 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_76', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_76', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_76(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 306)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ai/caij4wiud5l7spqmn33xlsvqtoeckzz2qjcnxy5qygdba5stwoah.py
# Topologically Sorted Source Nodes: [concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_46 => cat_45
#   concated_features_47 => cat_46
#   input_12 => cat_47
# Graph fragment:
#   %cat_45 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94], 1), kwargs = {})
#   %cat_46 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96], 1), kwargs = {})
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96, %convolution_98], 1), kwargs = {})
triton_poi_fused_cat_77 = async_compile.triton('triton_poi_fused_cat_77', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_77', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_77(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 81408*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 84480*x1), tmp0, None)
    tl.store(out_ptr2 + (x0 + 87552*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/lt/cltd3hzl6sb5ndjlylgfws7wicuabc6iog3suyyvjvirhosl25ho.py
# Topologically Sorted Source Nodes: [batch_norm_94, relu_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_94 => add_189, mul_283, mul_284, sub_94
#   relu_94 => relu_94
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_45, %unsqueeze_753), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_757), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_759), kwargs = {})
#   %relu_94 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_189,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_78 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_78', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_78', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_78(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 325632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 318)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6v/c6vcvzssfefufywkyoymsbyikta4s5lfw4n2vard76jywwj5zyhk.py
# Topologically Sorted Source Nodes: [concated_features_47, input_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concated_features_47 => cat_46
#   input_12 => cat_47
# Graph fragment:
#   %cat_46 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96], 1), kwargs = {})
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96, %convolution_98], 1), kwargs = {})
triton_poi_fused_cat_79 = async_compile.triton('triton_poi_fused_cat_79', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_79', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_79(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 84480*x1), tmp0, None)
    tl.store(out_ptr1 + (x0 + 87552*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/k4/ck4lqjncijmwyyrrvaxe4rex5isc4o3vybntzueubhzrp73nvsp6.py
# Topologically Sorted Source Nodes: [batch_norm_96, relu_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_96 => add_193, mul_289, mul_290, sub_96
#   relu_96 => relu_96
# Graph fragment:
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_46, %unsqueeze_769), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_773), kwargs = {})
#   %add_193 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_775), kwargs = {})
#   %relu_96 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_193,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_80 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_80', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_80', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_80(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 337920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 256) % 330)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6h/c6hyifvkymn5b7fsbx23rftrru5my6viuq4r5u6oiq3vc46rwop3.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_12 => cat_47
# Graph fragment:
#   %cat_47 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d_1, %convolution_68, %convolution_70, %convolution_72, %convolution_74, %convolution_76, %convolution_78, %convolution_80, %convolution_82, %convolution_84, %convolution_86, %convolution_88, %convolution_90, %convolution_92, %convolution_94, %convolution_96, %convolution_98], 1), kwargs = {})
triton_poi_fused_cat_81 = async_compile.triton('triton_poi_fused_cat_81', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_81', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_81(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 87552*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/3q/c3qeivk6e36s6hhz27clslrk4kjrvb2zsxj3iec7ny5hmmouwosk.py
# Topologically Sorted Source Nodes: [input_13, out, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   input_13 => add_197, mul_295, mul_296, sub_98
#   out => relu_98
#   out_1 => mean
# Graph fragment:
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_47, %unsqueeze_785), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %unsqueeze_787), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_295, %unsqueeze_789), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_296, %unsqueeze_791), kwargs = {})
#   %relu_98 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_197,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_98, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_82 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_82', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_82', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_82(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 1368
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 342)
    tmp0 = tl.load(in_ptr0 + (r2 + 256*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = 256.0
    tmp22 = tmp20 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_4, (24, ), (1, ))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, ), (1, ))
    assert_size_stride(primals_7, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_8, (48, ), (1, ))
    assert_size_stride(primals_9, (48, ), (1, ))
    assert_size_stride(primals_10, (48, ), (1, ))
    assert_size_stride(primals_11, (48, ), (1, ))
    assert_size_stride(primals_12, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_13, (36, ), (1, ))
    assert_size_stride(primals_14, (36, ), (1, ))
    assert_size_stride(primals_15, (36, ), (1, ))
    assert_size_stride(primals_16, (36, ), (1, ))
    assert_size_stride(primals_17, (48, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(primals_18, (48, ), (1, ))
    assert_size_stride(primals_19, (48, ), (1, ))
    assert_size_stride(primals_20, (48, ), (1, ))
    assert_size_stride(primals_21, (48, ), (1, ))
    assert_size_stride(primals_22, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_23, (48, ), (1, ))
    assert_size_stride(primals_24, (48, ), (1, ))
    assert_size_stride(primals_25, (48, ), (1, ))
    assert_size_stride(primals_26, (48, ), (1, ))
    assert_size_stride(primals_27, (48, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (48, ), (1, ))
    assert_size_stride(primals_30, (48, ), (1, ))
    assert_size_stride(primals_31, (48, ), (1, ))
    assert_size_stride(primals_32, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_33, (60, ), (1, ))
    assert_size_stride(primals_34, (60, ), (1, ))
    assert_size_stride(primals_35, (60, ), (1, ))
    assert_size_stride(primals_36, (60, ), (1, ))
    assert_size_stride(primals_37, (48, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_38, (48, ), (1, ))
    assert_size_stride(primals_39, (48, ), (1, ))
    assert_size_stride(primals_40, (48, ), (1, ))
    assert_size_stride(primals_41, (48, ), (1, ))
    assert_size_stride(primals_42, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_43, (72, ), (1, ))
    assert_size_stride(primals_44, (72, ), (1, ))
    assert_size_stride(primals_45, (72, ), (1, ))
    assert_size_stride(primals_46, (72, ), (1, ))
    assert_size_stride(primals_47, (48, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_48, (48, ), (1, ))
    assert_size_stride(primals_49, (48, ), (1, ))
    assert_size_stride(primals_50, (48, ), (1, ))
    assert_size_stride(primals_51, (48, ), (1, ))
    assert_size_stride(primals_52, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_53, (84, ), (1, ))
    assert_size_stride(primals_54, (84, ), (1, ))
    assert_size_stride(primals_55, (84, ), (1, ))
    assert_size_stride(primals_56, (84, ), (1, ))
    assert_size_stride(primals_57, (48, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(primals_58, (48, ), (1, ))
    assert_size_stride(primals_59, (48, ), (1, ))
    assert_size_stride(primals_60, (48, ), (1, ))
    assert_size_stride(primals_61, (48, ), (1, ))
    assert_size_stride(primals_62, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_63, (96, ), (1, ))
    assert_size_stride(primals_64, (96, ), (1, ))
    assert_size_stride(primals_65, (96, ), (1, ))
    assert_size_stride(primals_66, (96, ), (1, ))
    assert_size_stride(primals_67, (48, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_68, (48, ), (1, ))
    assert_size_stride(primals_69, (48, ), (1, ))
    assert_size_stride(primals_70, (48, ), (1, ))
    assert_size_stride(primals_71, (48, ), (1, ))
    assert_size_stride(primals_72, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_73, (108, ), (1, ))
    assert_size_stride(primals_74, (108, ), (1, ))
    assert_size_stride(primals_75, (108, ), (1, ))
    assert_size_stride(primals_76, (108, ), (1, ))
    assert_size_stride(primals_77, (48, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(primals_78, (48, ), (1, ))
    assert_size_stride(primals_79, (48, ), (1, ))
    assert_size_stride(primals_80, (48, ), (1, ))
    assert_size_stride(primals_81, (48, ), (1, ))
    assert_size_stride(primals_82, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_83, (120, ), (1, ))
    assert_size_stride(primals_84, (120, ), (1, ))
    assert_size_stride(primals_85, (120, ), (1, ))
    assert_size_stride(primals_86, (120, ), (1, ))
    assert_size_stride(primals_87, (48, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_88, (48, ), (1, ))
    assert_size_stride(primals_89, (48, ), (1, ))
    assert_size_stride(primals_90, (48, ), (1, ))
    assert_size_stride(primals_91, (48, ), (1, ))
    assert_size_stride(primals_92, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_93, (132, ), (1, ))
    assert_size_stride(primals_94, (132, ), (1, ))
    assert_size_stride(primals_95, (132, ), (1, ))
    assert_size_stride(primals_96, (132, ), (1, ))
    assert_size_stride(primals_97, (48, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_98, (48, ), (1, ))
    assert_size_stride(primals_99, (48, ), (1, ))
    assert_size_stride(primals_100, (48, ), (1, ))
    assert_size_stride(primals_101, (48, ), (1, ))
    assert_size_stride(primals_102, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_103, (144, ), (1, ))
    assert_size_stride(primals_104, (144, ), (1, ))
    assert_size_stride(primals_105, (144, ), (1, ))
    assert_size_stride(primals_106, (144, ), (1, ))
    assert_size_stride(primals_107, (48, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_108, (48, ), (1, ))
    assert_size_stride(primals_109, (48, ), (1, ))
    assert_size_stride(primals_110, (48, ), (1, ))
    assert_size_stride(primals_111, (48, ), (1, ))
    assert_size_stride(primals_112, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_113, (156, ), (1, ))
    assert_size_stride(primals_114, (156, ), (1, ))
    assert_size_stride(primals_115, (156, ), (1, ))
    assert_size_stride(primals_116, (156, ), (1, ))
    assert_size_stride(primals_117, (48, 156, 1, 1), (156, 1, 1, 1))
    assert_size_stride(primals_118, (48, ), (1, ))
    assert_size_stride(primals_119, (48, ), (1, ))
    assert_size_stride(primals_120, (48, ), (1, ))
    assert_size_stride(primals_121, (48, ), (1, ))
    assert_size_stride(primals_122, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_123, (168, ), (1, ))
    assert_size_stride(primals_124, (168, ), (1, ))
    assert_size_stride(primals_125, (168, ), (1, ))
    assert_size_stride(primals_126, (168, ), (1, ))
    assert_size_stride(primals_127, (48, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_128, (48, ), (1, ))
    assert_size_stride(primals_129, (48, ), (1, ))
    assert_size_stride(primals_130, (48, ), (1, ))
    assert_size_stride(primals_131, (48, ), (1, ))
    assert_size_stride(primals_132, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_133, (180, ), (1, ))
    assert_size_stride(primals_134, (180, ), (1, ))
    assert_size_stride(primals_135, (180, ), (1, ))
    assert_size_stride(primals_136, (180, ), (1, ))
    assert_size_stride(primals_137, (48, 180, 1, 1), (180, 1, 1, 1))
    assert_size_stride(primals_138, (48, ), (1, ))
    assert_size_stride(primals_139, (48, ), (1, ))
    assert_size_stride(primals_140, (48, ), (1, ))
    assert_size_stride(primals_141, (48, ), (1, ))
    assert_size_stride(primals_142, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_144, (192, ), (1, ))
    assert_size_stride(primals_145, (192, ), (1, ))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_148, (48, ), (1, ))
    assert_size_stride(primals_149, (48, ), (1, ))
    assert_size_stride(primals_150, (48, ), (1, ))
    assert_size_stride(primals_151, (48, ), (1, ))
    assert_size_stride(primals_152, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_153, (204, ), (1, ))
    assert_size_stride(primals_154, (204, ), (1, ))
    assert_size_stride(primals_155, (204, ), (1, ))
    assert_size_stride(primals_156, (204, ), (1, ))
    assert_size_stride(primals_157, (48, 204, 1, 1), (204, 1, 1, 1))
    assert_size_stride(primals_158, (48, ), (1, ))
    assert_size_stride(primals_159, (48, ), (1, ))
    assert_size_stride(primals_160, (48, ), (1, ))
    assert_size_stride(primals_161, (48, ), (1, ))
    assert_size_stride(primals_162, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_163, (216, ), (1, ))
    assert_size_stride(primals_164, (216, ), (1, ))
    assert_size_stride(primals_165, (216, ), (1, ))
    assert_size_stride(primals_166, (216, ), (1, ))
    assert_size_stride(primals_167, (108, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_168, (108, ), (1, ))
    assert_size_stride(primals_169, (108, ), (1, ))
    assert_size_stride(primals_170, (108, ), (1, ))
    assert_size_stride(primals_171, (108, ), (1, ))
    assert_size_stride(primals_172, (48, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(primals_173, (48, ), (1, ))
    assert_size_stride(primals_174, (48, ), (1, ))
    assert_size_stride(primals_175, (48, ), (1, ))
    assert_size_stride(primals_176, (48, ), (1, ))
    assert_size_stride(primals_177, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_178, (120, ), (1, ))
    assert_size_stride(primals_179, (120, ), (1, ))
    assert_size_stride(primals_180, (120, ), (1, ))
    assert_size_stride(primals_181, (120, ), (1, ))
    assert_size_stride(primals_182, (48, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_183, (48, ), (1, ))
    assert_size_stride(primals_184, (48, ), (1, ))
    assert_size_stride(primals_185, (48, ), (1, ))
    assert_size_stride(primals_186, (48, ), (1, ))
    assert_size_stride(primals_187, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_188, (132, ), (1, ))
    assert_size_stride(primals_189, (132, ), (1, ))
    assert_size_stride(primals_190, (132, ), (1, ))
    assert_size_stride(primals_191, (132, ), (1, ))
    assert_size_stride(primals_192, (48, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_193, (48, ), (1, ))
    assert_size_stride(primals_194, (48, ), (1, ))
    assert_size_stride(primals_195, (48, ), (1, ))
    assert_size_stride(primals_196, (48, ), (1, ))
    assert_size_stride(primals_197, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_198, (144, ), (1, ))
    assert_size_stride(primals_199, (144, ), (1, ))
    assert_size_stride(primals_200, (144, ), (1, ))
    assert_size_stride(primals_201, (144, ), (1, ))
    assert_size_stride(primals_202, (48, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_203, (48, ), (1, ))
    assert_size_stride(primals_204, (48, ), (1, ))
    assert_size_stride(primals_205, (48, ), (1, ))
    assert_size_stride(primals_206, (48, ), (1, ))
    assert_size_stride(primals_207, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_208, (156, ), (1, ))
    assert_size_stride(primals_209, (156, ), (1, ))
    assert_size_stride(primals_210, (156, ), (1, ))
    assert_size_stride(primals_211, (156, ), (1, ))
    assert_size_stride(primals_212, (48, 156, 1, 1), (156, 1, 1, 1))
    assert_size_stride(primals_213, (48, ), (1, ))
    assert_size_stride(primals_214, (48, ), (1, ))
    assert_size_stride(primals_215, (48, ), (1, ))
    assert_size_stride(primals_216, (48, ), (1, ))
    assert_size_stride(primals_217, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_218, (168, ), (1, ))
    assert_size_stride(primals_219, (168, ), (1, ))
    assert_size_stride(primals_220, (168, ), (1, ))
    assert_size_stride(primals_221, (168, ), (1, ))
    assert_size_stride(primals_222, (48, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_223, (48, ), (1, ))
    assert_size_stride(primals_224, (48, ), (1, ))
    assert_size_stride(primals_225, (48, ), (1, ))
    assert_size_stride(primals_226, (48, ), (1, ))
    assert_size_stride(primals_227, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_228, (180, ), (1, ))
    assert_size_stride(primals_229, (180, ), (1, ))
    assert_size_stride(primals_230, (180, ), (1, ))
    assert_size_stride(primals_231, (180, ), (1, ))
    assert_size_stride(primals_232, (48, 180, 1, 1), (180, 1, 1, 1))
    assert_size_stride(primals_233, (48, ), (1, ))
    assert_size_stride(primals_234, (48, ), (1, ))
    assert_size_stride(primals_235, (48, ), (1, ))
    assert_size_stride(primals_236, (48, ), (1, ))
    assert_size_stride(primals_237, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_238, (192, ), (1, ))
    assert_size_stride(primals_239, (192, ), (1, ))
    assert_size_stride(primals_240, (192, ), (1, ))
    assert_size_stride(primals_241, (192, ), (1, ))
    assert_size_stride(primals_242, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_243, (48, ), (1, ))
    assert_size_stride(primals_244, (48, ), (1, ))
    assert_size_stride(primals_245, (48, ), (1, ))
    assert_size_stride(primals_246, (48, ), (1, ))
    assert_size_stride(primals_247, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_248, (204, ), (1, ))
    assert_size_stride(primals_249, (204, ), (1, ))
    assert_size_stride(primals_250, (204, ), (1, ))
    assert_size_stride(primals_251, (204, ), (1, ))
    assert_size_stride(primals_252, (48, 204, 1, 1), (204, 1, 1, 1))
    assert_size_stride(primals_253, (48, ), (1, ))
    assert_size_stride(primals_254, (48, ), (1, ))
    assert_size_stride(primals_255, (48, ), (1, ))
    assert_size_stride(primals_256, (48, ), (1, ))
    assert_size_stride(primals_257, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_258, (216, ), (1, ))
    assert_size_stride(primals_259, (216, ), (1, ))
    assert_size_stride(primals_260, (216, ), (1, ))
    assert_size_stride(primals_261, (216, ), (1, ))
    assert_size_stride(primals_262, (48, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_263, (48, ), (1, ))
    assert_size_stride(primals_264, (48, ), (1, ))
    assert_size_stride(primals_265, (48, ), (1, ))
    assert_size_stride(primals_266, (48, ), (1, ))
    assert_size_stride(primals_267, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_268, (228, ), (1, ))
    assert_size_stride(primals_269, (228, ), (1, ))
    assert_size_stride(primals_270, (228, ), (1, ))
    assert_size_stride(primals_271, (228, ), (1, ))
    assert_size_stride(primals_272, (48, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(primals_273, (48, ), (1, ))
    assert_size_stride(primals_274, (48, ), (1, ))
    assert_size_stride(primals_275, (48, ), (1, ))
    assert_size_stride(primals_276, (48, ), (1, ))
    assert_size_stride(primals_277, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_278, (240, ), (1, ))
    assert_size_stride(primals_279, (240, ), (1, ))
    assert_size_stride(primals_280, (240, ), (1, ))
    assert_size_stride(primals_281, (240, ), (1, ))
    assert_size_stride(primals_282, (48, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_283, (48, ), (1, ))
    assert_size_stride(primals_284, (48, ), (1, ))
    assert_size_stride(primals_285, (48, ), (1, ))
    assert_size_stride(primals_286, (48, ), (1, ))
    assert_size_stride(primals_287, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_288, (252, ), (1, ))
    assert_size_stride(primals_289, (252, ), (1, ))
    assert_size_stride(primals_290, (252, ), (1, ))
    assert_size_stride(primals_291, (252, ), (1, ))
    assert_size_stride(primals_292, (48, 252, 1, 1), (252, 1, 1, 1))
    assert_size_stride(primals_293, (48, ), (1, ))
    assert_size_stride(primals_294, (48, ), (1, ))
    assert_size_stride(primals_295, (48, ), (1, ))
    assert_size_stride(primals_296, (48, ), (1, ))
    assert_size_stride(primals_297, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_298, (264, ), (1, ))
    assert_size_stride(primals_299, (264, ), (1, ))
    assert_size_stride(primals_300, (264, ), (1, ))
    assert_size_stride(primals_301, (264, ), (1, ))
    assert_size_stride(primals_302, (48, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_303, (48, ), (1, ))
    assert_size_stride(primals_304, (48, ), (1, ))
    assert_size_stride(primals_305, (48, ), (1, ))
    assert_size_stride(primals_306, (48, ), (1, ))
    assert_size_stride(primals_307, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_308, (276, ), (1, ))
    assert_size_stride(primals_309, (276, ), (1, ))
    assert_size_stride(primals_310, (276, ), (1, ))
    assert_size_stride(primals_311, (276, ), (1, ))
    assert_size_stride(primals_312, (48, 276, 1, 1), (276, 1, 1, 1))
    assert_size_stride(primals_313, (48, ), (1, ))
    assert_size_stride(primals_314, (48, ), (1, ))
    assert_size_stride(primals_315, (48, ), (1, ))
    assert_size_stride(primals_316, (48, ), (1, ))
    assert_size_stride(primals_317, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_318, (288, ), (1, ))
    assert_size_stride(primals_319, (288, ), (1, ))
    assert_size_stride(primals_320, (288, ), (1, ))
    assert_size_stride(primals_321, (288, ), (1, ))
    assert_size_stride(primals_322, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_323, (48, ), (1, ))
    assert_size_stride(primals_324, (48, ), (1, ))
    assert_size_stride(primals_325, (48, ), (1, ))
    assert_size_stride(primals_326, (48, ), (1, ))
    assert_size_stride(primals_327, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_328, (300, ), (1, ))
    assert_size_stride(primals_329, (300, ), (1, ))
    assert_size_stride(primals_330, (300, ), (1, ))
    assert_size_stride(primals_331, (300, ), (1, ))
    assert_size_stride(primals_332, (150, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(primals_333, (150, ), (1, ))
    assert_size_stride(primals_334, (150, ), (1, ))
    assert_size_stride(primals_335, (150, ), (1, ))
    assert_size_stride(primals_336, (150, ), (1, ))
    assert_size_stride(primals_337, (48, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_338, (48, ), (1, ))
    assert_size_stride(primals_339, (48, ), (1, ))
    assert_size_stride(primals_340, (48, ), (1, ))
    assert_size_stride(primals_341, (48, ), (1, ))
    assert_size_stride(primals_342, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_343, (162, ), (1, ))
    assert_size_stride(primals_344, (162, ), (1, ))
    assert_size_stride(primals_345, (162, ), (1, ))
    assert_size_stride(primals_346, (162, ), (1, ))
    assert_size_stride(primals_347, (48, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(primals_348, (48, ), (1, ))
    assert_size_stride(primals_349, (48, ), (1, ))
    assert_size_stride(primals_350, (48, ), (1, ))
    assert_size_stride(primals_351, (48, ), (1, ))
    assert_size_stride(primals_352, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_353, (174, ), (1, ))
    assert_size_stride(primals_354, (174, ), (1, ))
    assert_size_stride(primals_355, (174, ), (1, ))
    assert_size_stride(primals_356, (174, ), (1, ))
    assert_size_stride(primals_357, (48, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(primals_358, (48, ), (1, ))
    assert_size_stride(primals_359, (48, ), (1, ))
    assert_size_stride(primals_360, (48, ), (1, ))
    assert_size_stride(primals_361, (48, ), (1, ))
    assert_size_stride(primals_362, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_363, (186, ), (1, ))
    assert_size_stride(primals_364, (186, ), (1, ))
    assert_size_stride(primals_365, (186, ), (1, ))
    assert_size_stride(primals_366, (186, ), (1, ))
    assert_size_stride(primals_367, (48, 186, 1, 1), (186, 1, 1, 1))
    assert_size_stride(primals_368, (48, ), (1, ))
    assert_size_stride(primals_369, (48, ), (1, ))
    assert_size_stride(primals_370, (48, ), (1, ))
    assert_size_stride(primals_371, (48, ), (1, ))
    assert_size_stride(primals_372, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_373, (198, ), (1, ))
    assert_size_stride(primals_374, (198, ), (1, ))
    assert_size_stride(primals_375, (198, ), (1, ))
    assert_size_stride(primals_376, (198, ), (1, ))
    assert_size_stride(primals_377, (48, 198, 1, 1), (198, 1, 1, 1))
    assert_size_stride(primals_378, (48, ), (1, ))
    assert_size_stride(primals_379, (48, ), (1, ))
    assert_size_stride(primals_380, (48, ), (1, ))
    assert_size_stride(primals_381, (48, ), (1, ))
    assert_size_stride(primals_382, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_383, (210, ), (1, ))
    assert_size_stride(primals_384, (210, ), (1, ))
    assert_size_stride(primals_385, (210, ), (1, ))
    assert_size_stride(primals_386, (210, ), (1, ))
    assert_size_stride(primals_387, (48, 210, 1, 1), (210, 1, 1, 1))
    assert_size_stride(primals_388, (48, ), (1, ))
    assert_size_stride(primals_389, (48, ), (1, ))
    assert_size_stride(primals_390, (48, ), (1, ))
    assert_size_stride(primals_391, (48, ), (1, ))
    assert_size_stride(primals_392, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_393, (222, ), (1, ))
    assert_size_stride(primals_394, (222, ), (1, ))
    assert_size_stride(primals_395, (222, ), (1, ))
    assert_size_stride(primals_396, (222, ), (1, ))
    assert_size_stride(primals_397, (48, 222, 1, 1), (222, 1, 1, 1))
    assert_size_stride(primals_398, (48, ), (1, ))
    assert_size_stride(primals_399, (48, ), (1, ))
    assert_size_stride(primals_400, (48, ), (1, ))
    assert_size_stride(primals_401, (48, ), (1, ))
    assert_size_stride(primals_402, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_403, (234, ), (1, ))
    assert_size_stride(primals_404, (234, ), (1, ))
    assert_size_stride(primals_405, (234, ), (1, ))
    assert_size_stride(primals_406, (234, ), (1, ))
    assert_size_stride(primals_407, (48, 234, 1, 1), (234, 1, 1, 1))
    assert_size_stride(primals_408, (48, ), (1, ))
    assert_size_stride(primals_409, (48, ), (1, ))
    assert_size_stride(primals_410, (48, ), (1, ))
    assert_size_stride(primals_411, (48, ), (1, ))
    assert_size_stride(primals_412, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_413, (246, ), (1, ))
    assert_size_stride(primals_414, (246, ), (1, ))
    assert_size_stride(primals_415, (246, ), (1, ))
    assert_size_stride(primals_416, (246, ), (1, ))
    assert_size_stride(primals_417, (48, 246, 1, 1), (246, 1, 1, 1))
    assert_size_stride(primals_418, (48, ), (1, ))
    assert_size_stride(primals_419, (48, ), (1, ))
    assert_size_stride(primals_420, (48, ), (1, ))
    assert_size_stride(primals_421, (48, ), (1, ))
    assert_size_stride(primals_422, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_423, (258, ), (1, ))
    assert_size_stride(primals_424, (258, ), (1, ))
    assert_size_stride(primals_425, (258, ), (1, ))
    assert_size_stride(primals_426, (258, ), (1, ))
    assert_size_stride(primals_427, (48, 258, 1, 1), (258, 1, 1, 1))
    assert_size_stride(primals_428, (48, ), (1, ))
    assert_size_stride(primals_429, (48, ), (1, ))
    assert_size_stride(primals_430, (48, ), (1, ))
    assert_size_stride(primals_431, (48, ), (1, ))
    assert_size_stride(primals_432, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_433, (270, ), (1, ))
    assert_size_stride(primals_434, (270, ), (1, ))
    assert_size_stride(primals_435, (270, ), (1, ))
    assert_size_stride(primals_436, (270, ), (1, ))
    assert_size_stride(primals_437, (48, 270, 1, 1), (270, 1, 1, 1))
    assert_size_stride(primals_438, (48, ), (1, ))
    assert_size_stride(primals_439, (48, ), (1, ))
    assert_size_stride(primals_440, (48, ), (1, ))
    assert_size_stride(primals_441, (48, ), (1, ))
    assert_size_stride(primals_442, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_443, (282, ), (1, ))
    assert_size_stride(primals_444, (282, ), (1, ))
    assert_size_stride(primals_445, (282, ), (1, ))
    assert_size_stride(primals_446, (282, ), (1, ))
    assert_size_stride(primals_447, (48, 282, 1, 1), (282, 1, 1, 1))
    assert_size_stride(primals_448, (48, ), (1, ))
    assert_size_stride(primals_449, (48, ), (1, ))
    assert_size_stride(primals_450, (48, ), (1, ))
    assert_size_stride(primals_451, (48, ), (1, ))
    assert_size_stride(primals_452, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_453, (294, ), (1, ))
    assert_size_stride(primals_454, (294, ), (1, ))
    assert_size_stride(primals_455, (294, ), (1, ))
    assert_size_stride(primals_456, (294, ), (1, ))
    assert_size_stride(primals_457, (48, 294, 1, 1), (294, 1, 1, 1))
    assert_size_stride(primals_458, (48, ), (1, ))
    assert_size_stride(primals_459, (48, ), (1, ))
    assert_size_stride(primals_460, (48, ), (1, ))
    assert_size_stride(primals_461, (48, ), (1, ))
    assert_size_stride(primals_462, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_463, (306, ), (1, ))
    assert_size_stride(primals_464, (306, ), (1, ))
    assert_size_stride(primals_465, (306, ), (1, ))
    assert_size_stride(primals_466, (306, ), (1, ))
    assert_size_stride(primals_467, (48, 306, 1, 1), (306, 1, 1, 1))
    assert_size_stride(primals_468, (48, ), (1, ))
    assert_size_stride(primals_469, (48, ), (1, ))
    assert_size_stride(primals_470, (48, ), (1, ))
    assert_size_stride(primals_471, (48, ), (1, ))
    assert_size_stride(primals_472, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_473, (318, ), (1, ))
    assert_size_stride(primals_474, (318, ), (1, ))
    assert_size_stride(primals_475, (318, ), (1, ))
    assert_size_stride(primals_476, (318, ), (1, ))
    assert_size_stride(primals_477, (48, 318, 1, 1), (318, 1, 1, 1))
    assert_size_stride(primals_478, (48, ), (1, ))
    assert_size_stride(primals_479, (48, ), (1, ))
    assert_size_stride(primals_480, (48, ), (1, ))
    assert_size_stride(primals_481, (48, ), (1, ))
    assert_size_stride(primals_482, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_483, (330, ), (1, ))
    assert_size_stride(primals_484, (330, ), (1, ))
    assert_size_stride(primals_485, (330, ), (1, ))
    assert_size_stride(primals_486, (330, ), (1, ))
    assert_size_stride(primals_487, (48, 330, 1, 1), (330, 1, 1, 1))
    assert_size_stride(primals_488, (48, ), (1, ))
    assert_size_stride(primals_489, (48, ), (1, ))
    assert_size_stride(primals_490, (48, ), (1, ))
    assert_size_stride(primals_491, (48, ), (1, ))
    assert_size_stride(primals_492, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_493, (342, ), (1, ))
    assert_size_stride(primals_494, (342, ), (1, ))
    assert_size_stride(primals_495, (342, ), (1, ))
    assert_size_stride(primals_496, (342, ), (1, ))
    assert_size_stride(primals_497, (10, 342), (342, 1))
    assert_size_stride(primals_498, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 24, 64, 64), (98304, 4096, 64, 1))
        buf1 = empty_strided_cuda((4, 24, 64, 64), (98304, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 393216, grid=grid(393216), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [bottleneck_output], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf3 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_1, relu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf2, primals_8, primals_9, primals_10, primals_11, buf3, 786432, grid=grid(786432), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [new_features], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf5 = empty_strided_cuda((4, 36, 64, 64), (147456, 4096, 64, 1), torch.float32)
        buf6 = empty_strided_cuda((4, 36, 64, 64), (147456, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_1, batch_norm_2, relu_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2.run(buf0, buf4, primals_13, primals_14, primals_15, primals_16, buf5, buf6, 589824, grid=grid(589824), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [bottleneck_output_1], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf8 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf7, primals_18, primals_19, primals_20, primals_21, buf8, 786432, grid=grid(786432), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [new_features_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf10 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_2, batch_norm_4, relu_4], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_3.run(buf0, buf4, buf9, primals_23, primals_24, primals_25, primals_26, buf10, buf11, 786432, grid=grid(786432), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [bottleneck_output_2], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf13 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_5, relu_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf12, primals_28, primals_29, primals_30, primals_31, buf13, 786432, grid=grid(786432), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [new_features_2], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf15 = empty_strided_cuda((4, 60, 64, 64), (245760, 4096, 64, 1), torch.float32)
        buf16 = empty_strided_cuda((4, 60, 64, 64), (245760, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_3, batch_norm_6, relu_6], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4.run(buf0, buf4, buf9, buf14, primals_33, primals_34, primals_35, primals_36, buf15, buf16, 983040, grid=grid(983040), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [bottleneck_output_3], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf18 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_7, relu_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf17, primals_38, primals_39, primals_40, primals_41, buf18, 786432, grid=grid(786432), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [new_features_3], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf20 = empty_strided_cuda((4, 72, 64, 64), (294912, 4096, 64, 1), torch.float32)
        buf21 = empty_strided_cuda((4, 72, 64, 64), (294912, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_4, batch_norm_8, relu_8], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf0, buf4, buf9, buf14, buf19, primals_43, primals_44, primals_45, primals_46, buf20, buf21, 1179648, grid=grid(1179648), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [bottleneck_output_4], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf23 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_9, relu_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf22, primals_48, primals_49, primals_50, primals_51, buf23, 786432, grid=grid(786432), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [new_features_4], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf25 = empty_strided_cuda((4, 84, 64, 64), (344064, 4096, 64, 1), torch.float32)
        buf26 = empty_strided_cuda((4, 84, 64, 64), (344064, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_5, batch_norm_10, relu_10], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6.run(buf0, buf4, buf9, buf14, buf19, buf24, primals_53, primals_54, primals_55, primals_56, buf25, buf26, 1376256, grid=grid(1376256), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [bottleneck_output_5], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf28 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_11, relu_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf27, primals_58, primals_59, primals_60, primals_61, buf28, 786432, grid=grid(786432), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [new_features_5], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf30 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        buf31 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_6, batch_norm_12, relu_12], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7.run(buf0, buf4, buf9, buf14, buf19, buf24, buf29, primals_63, primals_64, primals_65, primals_66, buf30, buf31, 1572864, grid=grid(1572864), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [bottleneck_output_6], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf33 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_13, relu_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf32, primals_68, primals_69, primals_70, primals_71, buf33, 786432, grid=grid(786432), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [new_features_6], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf35 = empty_strided_cuda((4, 108, 64, 64), (442368, 4096, 64, 1), torch.float32)
        buf36 = empty_strided_cuda((4, 108, 64, 64), (442368, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_7, batch_norm_14, relu_14], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8.run(buf0, buf4, buf9, buf14, buf19, buf24, buf29, buf34, primals_73, primals_74, primals_75, primals_76, buf35, buf36, 1769472, grid=grid(1769472), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [bottleneck_output_7], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf38 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_15, relu_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf37, primals_78, primals_79, primals_80, primals_81, buf38, 786432, grid=grid(786432), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [new_features_7], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf49 = empty_strided_cuda((4, 120, 64, 64), (491520, 4096, 64, 1), torch.float32)
        buf40 = reinterpret_tensor(buf49, (4, 24, 64, 64), (491520, 4096, 64, 1), 0)  # alias
        buf64 = empty_strided_cuda((4, 132, 64, 64), (540672, 4096, 64, 1), torch.float32)
        buf54 = reinterpret_tensor(buf64, (4, 24, 64, 64), (540672, 4096, 64, 1), 0)  # alias
        buf80 = empty_strided_cuda((4, 144, 64, 64), (589824, 4096, 64, 1), torch.float32)
        buf69 = reinterpret_tensor(buf80, (4, 24, 64, 64), (589824, 4096, 64, 1), 0)  # alias
        buf97 = empty_strided_cuda((4, 156, 64, 64), (638976, 4096, 64, 1), torch.float32)
        buf85 = reinterpret_tensor(buf97, (4, 24, 64, 64), (638976, 4096, 64, 1), 0)  # alias
        buf115 = empty_strided_cuda((4, 168, 64, 64), (688128, 4096, 64, 1), torch.float32)
        buf102 = reinterpret_tensor(buf115, (4, 24, 64, 64), (688128, 4096, 64, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf0, buf40, buf54, buf69, buf85, buf102, 393216, grid=grid(393216), stream=stream0)
        buf41 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 98304)  # alias
        buf55 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 98304)  # alias
        buf70 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 98304)  # alias
        buf86 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 98304)  # alias
        buf103 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 98304)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf4, buf41, buf55, buf70, buf86, buf103, 196608, grid=grid(196608), stream=stream0)
        buf42 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 147456)  # alias
        buf56 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 147456)  # alias
        buf71 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 147456)  # alias
        buf87 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 147456)  # alias
        buf104 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 147456)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf9, buf42, buf56, buf71, buf87, buf104, 196608, grid=grid(196608), stream=stream0)
        buf43 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 196608)  # alias
        buf57 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 196608)  # alias
        buf72 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 196608)  # alias
        buf88 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 196608)  # alias
        buf105 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 196608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf14, buf43, buf57, buf72, buf88, buf105, 196608, grid=grid(196608), stream=stream0)
        buf44 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 245760)  # alias
        buf58 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 245760)  # alias
        buf73 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 245760)  # alias
        buf89 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 245760)  # alias
        buf106 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 245760)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf19, buf44, buf58, buf73, buf89, buf106, 196608, grid=grid(196608), stream=stream0)
        buf45 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 294912)  # alias
        buf59 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 294912)  # alias
        buf74 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 294912)  # alias
        buf90 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 294912)  # alias
        buf107 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 294912)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf24, buf45, buf59, buf74, buf90, buf107, 196608, grid=grid(196608), stream=stream0)
        buf46 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 344064)  # alias
        buf60 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 344064)  # alias
        buf75 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 344064)  # alias
        buf91 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 344064)  # alias
        buf108 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 344064)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf29, buf46, buf60, buf75, buf91, buf108, 196608, grid=grid(196608), stream=stream0)
        buf47 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 393216)  # alias
        buf61 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 393216)  # alias
        buf76 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 393216)  # alias
        buf92 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 393216)  # alias
        buf109 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 393216)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf34, buf47, buf61, buf76, buf92, buf109, 196608, grid=grid(196608), stream=stream0)
        buf48 = reinterpret_tensor(buf49, (4, 12, 64, 64), (491520, 4096, 64, 1), 442368)  # alias
        buf62 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 442368)  # alias
        buf77 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 442368)  # alias
        buf93 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 442368)  # alias
        buf110 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 442368)  # alias
        # Topologically Sorted Source Nodes: [concated_features_8, concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf39, buf48, buf62, buf77, buf93, buf110, 196608, grid=grid(196608), stream=stream0)
        buf50 = empty_strided_cuda((4, 120, 64, 64), (491520, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_16, relu_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf49, primals_83, primals_84, primals_85, primals_86, buf50, 1966080, grid=grid(1966080), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [bottleneck_output_8], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf52 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_17, relu_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf51, primals_88, primals_89, primals_90, primals_91, buf52, 786432, grid=grid(786432), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [new_features_8], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf63 = reinterpret_tensor(buf64, (4, 12, 64, 64), (540672, 4096, 64, 1), 491520)  # alias
        buf78 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 491520)  # alias
        buf94 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 491520)  # alias
        buf111 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 491520)  # alias
        # Topologically Sorted Source Nodes: [concated_features_9, concated_features_10, concated_features_11, concated_features_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf53, buf63, buf78, buf94, buf111, 196608, grid=grid(196608), stream=stream0)
        buf65 = empty_strided_cuda((4, 132, 64, 64), (540672, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_18, relu_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf64, primals_93, primals_94, primals_95, primals_96, buf65, 2162688, grid=grid(2162688), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [bottleneck_output_9], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf67 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_19, relu_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf66, primals_98, primals_99, primals_100, primals_101, buf67, 786432, grid=grid(786432), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [new_features_9], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf79 = reinterpret_tensor(buf80, (4, 12, 64, 64), (589824, 4096, 64, 1), 540672)  # alias
        buf95 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 540672)  # alias
        buf112 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 540672)  # alias
        buf134 = empty_strided_cuda((4, 180, 64, 64), (737280, 4096, 64, 1), torch.float32)
        buf130 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 540672)  # alias
        # Topologically Sorted Source Nodes: [concated_features_10, concated_features_11, concated_features_12, concated_features_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf68, buf79, buf95, buf112, buf130, 196608, grid=grid(196608), stream=stream0)
        buf81 = empty_strided_cuda((4, 144, 64, 64), (589824, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_20, relu_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf80, primals_103, primals_104, primals_105, primals_106, buf81, 2359296, grid=grid(2359296), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [bottleneck_output_10], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf83 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_21, relu_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf82, primals_108, primals_109, primals_110, primals_111, buf83, 786432, grid=grid(786432), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [new_features_10], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf96 = reinterpret_tensor(buf97, (4, 12, 64, 64), (638976, 4096, 64, 1), 589824)  # alias
        buf113 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 589824)  # alias
        buf131 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 589824)  # alias
        buf154 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        buf150 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 589824)  # alias
        # Topologically Sorted Source Nodes: [concated_features_11, concated_features_12, concated_features_13, concated_features_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf84, buf96, buf113, buf131, buf150, 196608, grid=grid(196608), stream=stream0)
        buf98 = empty_strided_cuda((4, 156, 64, 64), (638976, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_22, relu_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf97, primals_113, primals_114, primals_115, primals_116, buf98, 2555904, grid=grid(2555904), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [bottleneck_output_11], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf100 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_23, relu_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf99, primals_118, primals_119, primals_120, primals_121, buf100, 786432, grid=grid(786432), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [new_features_11], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf114 = reinterpret_tensor(buf115, (4, 12, 64, 64), (688128, 4096, 64, 1), 638976)  # alias
        buf132 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 638976)  # alias
        buf151 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 638976)  # alias
        buf175 = empty_strided_cuda((4, 204, 64, 64), (835584, 4096, 64, 1), torch.float32)
        buf171 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 638976)  # alias
        # Topologically Sorted Source Nodes: [concated_features_12, concated_features_13, concated_features_14, concated_features_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf101, buf114, buf132, buf151, buf171, 196608, grid=grid(196608), stream=stream0)
        buf116 = empty_strided_cuda((4, 168, 64, 64), (688128, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_24, relu_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf115, primals_123, primals_124, primals_125, primals_126, buf116, 2752512, grid=grid(2752512), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [bottleneck_output_12], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf118 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_25, relu_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf117, primals_128, primals_129, primals_130, primals_131, buf118, 786432, grid=grid(786432), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [new_features_12], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf120 = reinterpret_tensor(buf134, (4, 24, 64, 64), (737280, 4096, 64, 1), 0)  # alias
        buf139 = reinterpret_tensor(buf154, (4, 24, 64, 64), (786432, 4096, 64, 1), 0)  # alias
        buf159 = reinterpret_tensor(buf175, (4, 24, 64, 64), (835584, 4096, 64, 1), 0)  # alias
        buf197 = empty_strided_cuda((4, 216, 64, 64), (884736, 4096, 64, 1), torch.float32)
        buf180 = reinterpret_tensor(buf197, (4, 24, 64, 64), (884736, 4096, 64, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf0, buf120, buf139, buf159, buf180, 393216, grid=grid(393216), stream=stream0)
        buf121 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 98304)  # alias
        buf140 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 98304)  # alias
        buf160 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 98304)  # alias
        buf181 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 98304)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf4, buf121, buf140, buf160, buf181, 196608, grid=grid(196608), stream=stream0)
        buf122 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 147456)  # alias
        buf141 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 147456)  # alias
        buf161 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 147456)  # alias
        buf182 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 147456)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf9, buf122, buf141, buf161, buf182, 196608, grid=grid(196608), stream=stream0)
        buf123 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 196608)  # alias
        buf142 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 196608)  # alias
        buf162 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 196608)  # alias
        buf183 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 196608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf14, buf123, buf142, buf162, buf183, 196608, grid=grid(196608), stream=stream0)
        buf124 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 245760)  # alias
        buf143 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 245760)  # alias
        buf163 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 245760)  # alias
        buf184 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 245760)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf19, buf124, buf143, buf163, buf184, 196608, grid=grid(196608), stream=stream0)
        buf125 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 294912)  # alias
        buf144 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 294912)  # alias
        buf164 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 294912)  # alias
        buf185 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 294912)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf24, buf125, buf144, buf164, buf185, 196608, grid=grid(196608), stream=stream0)
        buf126 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 344064)  # alias
        buf145 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 344064)  # alias
        buf165 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 344064)  # alias
        buf186 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 344064)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf29, buf126, buf145, buf165, buf186, 196608, grid=grid(196608), stream=stream0)
        buf127 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 393216)  # alias
        buf146 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 393216)  # alias
        buf166 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 393216)  # alias
        buf187 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 393216)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf34, buf127, buf146, buf166, buf187, 196608, grid=grid(196608), stream=stream0)
        buf128 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 442368)  # alias
        buf147 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 442368)  # alias
        buf167 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 442368)  # alias
        buf188 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 442368)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf39, buf128, buf147, buf167, buf188, 196608, grid=grid(196608), stream=stream0)
        buf129 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 491520)  # alias
        buf148 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 491520)  # alias
        buf168 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 491520)  # alias
        buf189 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 491520)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf53, buf129, buf148, buf168, buf189, 196608, grid=grid(196608), stream=stream0)
        buf133 = reinterpret_tensor(buf134, (4, 12, 64, 64), (737280, 4096, 64, 1), 688128)  # alias
        buf152 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 688128)  # alias
        buf172 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 688128)  # alias
        buf193 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 688128)  # alias
        # Topologically Sorted Source Nodes: [concated_features_13, concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf119, buf133, buf152, buf172, buf193, 196608, grid=grid(196608), stream=stream0)
        buf135 = empty_strided_cuda((4, 180, 64, 64), (737280, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_26, relu_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf134, primals_133, primals_134, primals_135, primals_136, buf135, 2949120, grid=grid(2949120), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [bottleneck_output_13], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf137 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_27, relu_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf136, primals_138, primals_139, primals_140, primals_141, buf137, 786432, grid=grid(786432), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [new_features_13], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf149 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 540672)  # alias
        buf169 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 540672)  # alias
        buf190 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 540672)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf68, buf149, buf169, buf190, 196608, grid=grid(196608), stream=stream0)
        buf153 = reinterpret_tensor(buf154, (4, 12, 64, 64), (786432, 4096, 64, 1), 737280)  # alias
        buf173 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 737280)  # alias
        buf194 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 737280)  # alias
        # Topologically Sorted Source Nodes: [concated_features_14, concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf138, buf153, buf173, buf194, 196608, grid=grid(196608), stream=stream0)
        buf155 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_28, relu_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf154, primals_143, primals_144, primals_145, primals_146, buf155, 3145728, grid=grid(3145728), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [bottleneck_output_14], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf157 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_29, relu_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf156, primals_148, primals_149, primals_150, primals_151, buf157, 786432, grid=grid(786432), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [new_features_14], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf170 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 589824)  # alias
        buf191 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 589824)  # alias
        # Topologically Sorted Source Nodes: [concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf84, buf170, buf191, 196608, grid=grid(196608), stream=stream0)
        buf174 = reinterpret_tensor(buf175, (4, 12, 64, 64), (835584, 4096, 64, 1), 786432)  # alias
        buf195 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 786432)  # alias
        # Topologically Sorted Source Nodes: [concated_features_15, input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf158, buf174, buf195, 196608, grid=grid(196608), stream=stream0)
        buf176 = empty_strided_cuda((4, 204, 64, 64), (835584, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_30, relu_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf175, primals_153, primals_154, primals_155, primals_156, buf176, 3342336, grid=grid(3342336), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [bottleneck_output_15], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 48, 64, 64), (196608, 4096, 64, 1))
        buf178 = empty_strided_cuda((4, 48, 64, 64), (196608, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_31, relu_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf177, primals_158, primals_159, primals_160, primals_161, buf178, 786432, grid=grid(786432), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [new_features_15], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 12, 64, 64), (49152, 4096, 64, 1))
        buf192 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 638976)  # alias
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_27.run(buf101, buf192, 196608, grid=grid(196608), stream=stream0)
        buf196 = reinterpret_tensor(buf197, (4, 12, 64, 64), (884736, 4096, 64, 1), 835584)  # alias
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_27.run(buf179, buf196, 196608, grid=grid(196608), stream=stream0)
        buf198 = empty_strided_cuda((4, 216, 64, 64), (884736, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf197, primals_163, primals_164, primals_165, primals_166, buf198, 3538944, grid=grid(3538944), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 108, 64, 64), (442368, 4096, 64, 1))
        buf200 = empty_strided_cuda((4, 108, 32, 32), (110592, 1024, 32, 1), torch.float32)
        buf201 = empty_strided_cuda((4, 108, 32, 32), (110592, 1024, 32, 1), torch.float32)
        buf249 = empty_strided_cuda((4, 204, 32, 32), (208896, 1024, 32, 1), torch.float32)
        buf240 = reinterpret_tensor(buf249, (4, 108, 32, 32), (208896, 1024, 32, 1), 0)  # alias
        buf264 = empty_strided_cuda((4, 216, 32, 32), (221184, 1024, 32, 1), torch.float32)
        buf254 = reinterpret_tensor(buf264, (4, 108, 32, 32), (221184, 1024, 32, 1), 0)  # alias
        buf280 = empty_strided_cuda((4, 228, 32, 32), (233472, 1024, 32, 1), torch.float32)
        buf269 = reinterpret_tensor(buf280, (4, 108, 32, 32), (233472, 1024, 32, 1), 0)  # alias
        buf297 = empty_strided_cuda((4, 240, 32, 32), (245760, 1024, 32, 1), torch.float32)
        buf285 = reinterpret_tensor(buf297, (4, 108, 32, 32), (245760, 1024, 32, 1), 0)  # alias
        buf315 = empty_strided_cuda((4, 252, 32, 32), (258048, 1024, 32, 1), torch.float32)
        buf302 = reinterpret_tensor(buf315, (4, 108, 32, 32), (258048, 1024, 32, 1), 0)  # alias
        buf334 = empty_strided_cuda((4, 264, 32, 32), (270336, 1024, 32, 1), torch.float32)
        buf320 = reinterpret_tensor(buf334, (4, 108, 32, 32), (270336, 1024, 32, 1), 0)  # alias
        buf354 = empty_strided_cuda((4, 276, 32, 32), (282624, 1024, 32, 1), torch.float32)
        buf339 = reinterpret_tensor(buf354, (4, 108, 32, 32), (282624, 1024, 32, 1), 0)  # alias
        buf375 = empty_strided_cuda((4, 288, 32, 32), (294912, 1024, 32, 1), torch.float32)
        buf359 = reinterpret_tensor(buf375, (4, 108, 32, 32), (294912, 1024, 32, 1), 0)  # alias
        buf397 = empty_strided_cuda((4, 300, 32, 32), (307200, 1024, 32, 1), torch.float32)
        buf380 = reinterpret_tensor(buf397, (4, 108, 32, 32), (307200, 1024, 32, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_6, batch_norm_33, relu_33, concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28, concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29.run(buf199, primals_168, primals_169, primals_170, primals_171, buf200, buf201, buf240, buf254, buf269, buf285, buf302, buf320, buf339, buf359, buf380, 442368, grid=grid(442368), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [bottleneck_output_16], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf203 = reinterpret_tensor(buf179, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_34, relu_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf202, primals_173, primals_174, primals_175, primals_176, buf203, 196608, grid=grid(196608), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [new_features_16], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf205 = empty_strided_cuda((4, 120, 32, 32), (122880, 1024, 32, 1), torch.float32)
        buf206 = empty_strided_cuda((4, 120, 32, 32), (122880, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_17, batch_norm_35, relu_35], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31.run(buf200, buf204, primals_178, primals_179, primals_180, primals_181, buf205, buf206, 491520, grid=grid(491520), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [bottleneck_output_17], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf208 = reinterpret_tensor(buf101, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_36, relu_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf207, primals_183, primals_184, primals_185, primals_186, buf208, 196608, grid=grid(196608), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [new_features_17], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf210 = empty_strided_cuda((4, 132, 32, 32), (135168, 1024, 32, 1), torch.float32)
        buf211 = empty_strided_cuda((4, 132, 32, 32), (135168, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_18, batch_norm_37, relu_37], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32.run(buf200, buf204, buf209, primals_188, primals_189, primals_190, primals_191, buf210, buf211, 540672, grid=grid(540672), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [bottleneck_output_18], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf213 = reinterpret_tensor(buf158, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_38, relu_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf212, primals_193, primals_194, primals_195, primals_196, buf213, 196608, grid=grid(196608), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [new_features_18], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf215 = empty_strided_cuda((4, 144, 32, 32), (147456, 1024, 32, 1), torch.float32)
        buf216 = empty_strided_cuda((4, 144, 32, 32), (147456, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_19, batch_norm_39, relu_39], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33.run(buf200, buf204, buf209, buf214, primals_198, primals_199, primals_200, primals_201, buf215, buf216, 589824, grid=grid(589824), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [bottleneck_output_19], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf218 = reinterpret_tensor(buf84, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_40, relu_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf217, primals_203, primals_204, primals_205, primals_206, buf218, 196608, grid=grid(196608), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [new_features_19], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf220 = empty_strided_cuda((4, 156, 32, 32), (159744, 1024, 32, 1), torch.float32)
        buf221 = empty_strided_cuda((4, 156, 32, 32), (159744, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_20, batch_norm_41, relu_41], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34.run(buf200, buf204, buf209, buf214, buf219, primals_208, primals_209, primals_210, primals_211, buf220, buf221, 638976, grid=grid(638976), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [bottleneck_output_20], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf223 = reinterpret_tensor(buf138, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_42, relu_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf222, primals_213, primals_214, primals_215, primals_216, buf223, 196608, grid=grid(196608), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [new_features_20], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf225 = empty_strided_cuda((4, 168, 32, 32), (172032, 1024, 32, 1), torch.float32)
        buf226 = empty_strided_cuda((4, 168, 32, 32), (172032, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_21, batch_norm_43, relu_43], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35.run(buf200, buf204, buf209, buf214, buf219, buf224, primals_218, primals_219, primals_220, primals_221, buf225, buf226, 688128, grid=grid(688128), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [bottleneck_output_21], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf228 = reinterpret_tensor(buf68, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_44, relu_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf227, primals_223, primals_224, primals_225, primals_226, buf228, 196608, grid=grid(196608), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [new_features_21], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf230 = empty_strided_cuda((4, 180, 32, 32), (184320, 1024, 32, 1), torch.float32)
        buf231 = empty_strided_cuda((4, 180, 32, 32), (184320, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_22, batch_norm_45, relu_45], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36.run(buf200, buf204, buf209, buf214, buf219, buf224, buf229, primals_228, primals_229, primals_230, primals_231, buf230, buf231, 737280, grid=grid(737280), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [bottleneck_output_22], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf233 = reinterpret_tensor(buf119, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_46, relu_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf232, primals_233, primals_234, primals_235, primals_236, buf233, 196608, grid=grid(196608), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [new_features_22], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf235 = empty_strided_cuda((4, 192, 32, 32), (196608, 1024, 32, 1), torch.float32)
        buf236 = empty_strided_cuda((4, 192, 32, 32), (196608, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_23, batch_norm_47, relu_47], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37.run(buf200, buf204, buf209, buf214, buf219, buf224, buf229, buf234, primals_238, primals_239, primals_240, primals_241, buf235, buf236, 786432, grid=grid(786432), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [bottleneck_output_23], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf238 = reinterpret_tensor(buf53, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_48, relu_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf237, primals_243, primals_244, primals_245, primals_246, buf238, 196608, grid=grid(196608), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [new_features_23], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf241 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 110592)  # alias
        buf255 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 110592)  # alias
        buf270 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 110592)  # alias
        buf286 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 110592)  # alias
        buf303 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 110592)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf204, buf241, buf255, buf270, buf286, buf303, 49152, grid=grid(49152), stream=stream0)
        buf242 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 122880)  # alias
        buf256 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 122880)  # alias
        buf271 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 122880)  # alias
        buf287 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 122880)  # alias
        buf304 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 122880)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf209, buf242, buf256, buf271, buf287, buf304, 49152, grid=grid(49152), stream=stream0)
        buf243 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 135168)  # alias
        buf257 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 135168)  # alias
        buf272 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 135168)  # alias
        buf288 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 135168)  # alias
        buf305 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 135168)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf214, buf243, buf257, buf272, buf288, buf305, 49152, grid=grid(49152), stream=stream0)
        buf244 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 147456)  # alias
        buf258 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 147456)  # alias
        buf273 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 147456)  # alias
        buf289 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 147456)  # alias
        buf306 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 147456)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf219, buf244, buf258, buf273, buf289, buf306, 49152, grid=grid(49152), stream=stream0)
        buf245 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 159744)  # alias
        buf259 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 159744)  # alias
        buf274 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 159744)  # alias
        buf290 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 159744)  # alias
        buf307 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 159744)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf224, buf245, buf259, buf274, buf290, buf307, 49152, grid=grid(49152), stream=stream0)
        buf246 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 172032)  # alias
        buf260 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 172032)  # alias
        buf275 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 172032)  # alias
        buf291 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 172032)  # alias
        buf308 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 172032)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf229, buf246, buf260, buf275, buf291, buf308, 49152, grid=grid(49152), stream=stream0)
        buf247 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 184320)  # alias
        buf261 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 184320)  # alias
        buf276 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 184320)  # alias
        buf292 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 184320)  # alias
        buf309 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 184320)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf234, buf247, buf261, buf276, buf292, buf309, 49152, grid=grid(49152), stream=stream0)
        buf248 = reinterpret_tensor(buf249, (4, 12, 32, 32), (208896, 1024, 32, 1), 196608)  # alias
        buf262 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 196608)  # alias
        buf277 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 196608)  # alias
        buf293 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 196608)  # alias
        buf310 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 196608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_24, concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_38.run(buf239, buf248, buf262, buf277, buf293, buf310, 49152, grid=grid(49152), stream=stream0)
        buf250 = empty_strided_cuda((4, 204, 32, 32), (208896, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_49, relu_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_39.run(buf249, primals_248, primals_249, primals_250, primals_251, buf250, 835584, grid=grid(835584), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [bottleneck_output_24], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf252 = reinterpret_tensor(buf39, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_50, relu_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf251, primals_253, primals_254, primals_255, primals_256, buf252, 196608, grid=grid(196608), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [new_features_24], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf263 = reinterpret_tensor(buf264, (4, 12, 32, 32), (221184, 1024, 32, 1), 208896)  # alias
        buf278 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 208896)  # alias
        buf294 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 208896)  # alias
        buf311 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 208896)  # alias
        # Topologically Sorted Source Nodes: [concated_features_25, concated_features_26, concated_features_27, concated_features_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_40.run(buf253, buf263, buf278, buf294, buf311, 49152, grid=grid(49152), stream=stream0)
        buf265 = empty_strided_cuda((4, 216, 32, 32), (221184, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_51, relu_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf264, primals_258, primals_259, primals_260, primals_261, buf265, 884736, grid=grid(884736), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [bottleneck_output_25], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf267 = reinterpret_tensor(buf34, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_52, relu_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf266, primals_263, primals_264, primals_265, primals_266, buf267, 196608, grid=grid(196608), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [new_features_25], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf279 = reinterpret_tensor(buf280, (4, 12, 32, 32), (233472, 1024, 32, 1), 221184)  # alias
        buf295 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 221184)  # alias
        buf312 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 221184)  # alias
        buf330 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 221184)  # alias
        # Topologically Sorted Source Nodes: [concated_features_26, concated_features_27, concated_features_28, concated_features_29], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_42.run(buf268, buf279, buf295, buf312, buf330, 49152, grid=grid(49152), stream=stream0)
        buf281 = empty_strided_cuda((4, 228, 32, 32), (233472, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_53, relu_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf280, primals_268, primals_269, primals_270, primals_271, buf281, 933888, grid=grid(933888), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [bottleneck_output_26], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf283 = reinterpret_tensor(buf29, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_54, relu_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf282, primals_273, primals_274, primals_275, primals_276, buf283, 196608, grid=grid(196608), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [new_features_26], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf296 = reinterpret_tensor(buf297, (4, 12, 32, 32), (245760, 1024, 32, 1), 233472)  # alias
        buf313 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 233472)  # alias
        buf331 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 233472)  # alias
        buf350 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 233472)  # alias
        # Topologically Sorted Source Nodes: [concated_features_27, concated_features_28, concated_features_29, concated_features_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_44.run(buf284, buf296, buf313, buf331, buf350, 49152, grid=grid(49152), stream=stream0)
        buf298 = empty_strided_cuda((4, 240, 32, 32), (245760, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_55, relu_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf297, primals_278, primals_279, primals_280, primals_281, buf298, 983040, grid=grid(983040), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [bottleneck_output_27], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf300 = reinterpret_tensor(buf24, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_56, relu_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf299, primals_283, primals_284, primals_285, primals_286, buf300, 196608, grid=grid(196608), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [new_features_27], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf314 = reinterpret_tensor(buf315, (4, 12, 32, 32), (258048, 1024, 32, 1), 245760)  # alias
        buf332 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 245760)  # alias
        buf351 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 245760)  # alias
        buf371 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 245760)  # alias
        # Topologically Sorted Source Nodes: [concated_features_28, concated_features_29, concated_features_30, concated_features_31], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_46.run(buf301, buf314, buf332, buf351, buf371, 49152, grid=grid(49152), stream=stream0)
        buf316 = empty_strided_cuda((4, 252, 32, 32), (258048, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_57, relu_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf315, primals_288, primals_289, primals_290, primals_291, buf316, 1032192, grid=grid(1032192), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [bottleneck_output_28], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf318 = reinterpret_tensor(buf19, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_58, relu_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf317, primals_293, primals_294, primals_295, primals_296, buf318, 196608, grid=grid(196608), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [new_features_28], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf321 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 110592)  # alias
        buf340 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 110592)  # alias
        buf360 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 110592)  # alias
        buf381 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 110592)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf204, buf321, buf340, buf360, buf381, 49152, grid=grid(49152), stream=stream0)
        buf322 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 122880)  # alias
        buf341 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 122880)  # alias
        buf361 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 122880)  # alias
        buf382 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 122880)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf209, buf322, buf341, buf361, buf382, 49152, grid=grid(49152), stream=stream0)
        buf323 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 135168)  # alias
        buf342 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 135168)  # alias
        buf362 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 135168)  # alias
        buf383 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 135168)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf214, buf323, buf342, buf362, buf383, 49152, grid=grid(49152), stream=stream0)
        buf324 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 147456)  # alias
        buf343 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 147456)  # alias
        buf363 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 147456)  # alias
        buf384 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 147456)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf219, buf324, buf343, buf363, buf384, 49152, grid=grid(49152), stream=stream0)
        buf325 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 159744)  # alias
        buf344 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 159744)  # alias
        buf364 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 159744)  # alias
        buf385 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 159744)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf224, buf325, buf344, buf364, buf385, 49152, grid=grid(49152), stream=stream0)
        buf326 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 172032)  # alias
        buf345 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 172032)  # alias
        buf365 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 172032)  # alias
        buf386 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 172032)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf229, buf326, buf345, buf365, buf386, 49152, grid=grid(49152), stream=stream0)
        buf327 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 184320)  # alias
        buf346 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 184320)  # alias
        buf366 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 184320)  # alias
        buf387 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 184320)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf234, buf327, buf346, buf366, buf387, 49152, grid=grid(49152), stream=stream0)
        buf328 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 196608)  # alias
        buf347 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 196608)  # alias
        buf367 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 196608)  # alias
        buf388 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 196608)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf239, buf328, buf347, buf367, buf388, 49152, grid=grid(49152), stream=stream0)
        buf329 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 208896)  # alias
        buf348 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 208896)  # alias
        buf368 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 208896)  # alias
        buf389 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 208896)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf253, buf329, buf348, buf368, buf389, 49152, grid=grid(49152), stream=stream0)
        buf333 = reinterpret_tensor(buf334, (4, 12, 32, 32), (270336, 1024, 32, 1), 258048)  # alias
        buf352 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 258048)  # alias
        buf372 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 258048)  # alias
        buf393 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 258048)  # alias
        # Topologically Sorted Source Nodes: [concated_features_29, concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf319, buf333, buf352, buf372, buf393, 49152, grid=grid(49152), stream=stream0)
        buf335 = empty_strided_cuda((4, 264, 32, 32), (270336, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_59, relu_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf334, primals_298, primals_299, primals_300, primals_301, buf335, 1081344, grid=grid(1081344), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [bottleneck_output_29], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf337 = reinterpret_tensor(buf14, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_60, relu_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf336, primals_303, primals_304, primals_305, primals_306, buf337, 196608, grid=grid(196608), stream=stream0)
        del primals_306
        # Topologically Sorted Source Nodes: [new_features_29], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, primals_307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf349 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 221184)  # alias
        buf369 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 221184)  # alias
        buf390 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 221184)  # alias
        # Topologically Sorted Source Nodes: [concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_50.run(buf268, buf349, buf369, buf390, 49152, grid=grid(49152), stream=stream0)
        buf353 = reinterpret_tensor(buf354, (4, 12, 32, 32), (282624, 1024, 32, 1), 270336)  # alias
        buf373 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 270336)  # alias
        buf394 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 270336)  # alias
        # Topologically Sorted Source Nodes: [concated_features_30, concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_50.run(buf338, buf353, buf373, buf394, 49152, grid=grid(49152), stream=stream0)
        buf355 = empty_strided_cuda((4, 276, 32, 32), (282624, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_61, relu_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf354, primals_308, primals_309, primals_310, primals_311, buf355, 1130496, grid=grid(1130496), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [bottleneck_output_30], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf357 = reinterpret_tensor(buf9, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_62, relu_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf356, primals_313, primals_314, primals_315, primals_316, buf357, 196608, grid=grid(196608), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [new_features_30], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf370 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 233472)  # alias
        buf391 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 233472)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf284, buf370, buf391, 49152, grid=grid(49152), stream=stream0)
        buf374 = reinterpret_tensor(buf375, (4, 12, 32, 32), (294912, 1024, 32, 1), 282624)  # alias
        buf395 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 282624)  # alias
        # Topologically Sorted Source Nodes: [concated_features_31, input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_52.run(buf358, buf374, buf395, 49152, grid=grid(49152), stream=stream0)
        buf376 = empty_strided_cuda((4, 288, 32, 32), (294912, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_63, relu_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_53.run(buf375, primals_318, primals_319, primals_320, primals_321, buf376, 1179648, grid=grid(1179648), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [bottleneck_output_31], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 48, 32, 32), (49152, 1024, 32, 1))
        buf378 = reinterpret_tensor(buf4, (4, 48, 32, 32), (49152, 1024, 32, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_64, relu_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf377, primals_323, primals_324, primals_325, primals_326, buf378, 196608, grid=grid(196608), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [new_features_31], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 12, 32, 32), (12288, 1024, 32, 1))
        buf392 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 245760)  # alias
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf301, buf392, 49152, grid=grid(49152), stream=stream0)
        buf396 = reinterpret_tensor(buf397, (4, 12, 32, 32), (307200, 1024, 32, 1), 294912)  # alias
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf379, buf396, 49152, grid=grid(49152), stream=stream0)
        buf398 = empty_strided_cuda((4, 300, 32, 32), (307200, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf397, primals_328, primals_329, primals_330, primals_331, buf398, 1228800, grid=grid(1228800), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (4, 150, 32, 32), (153600, 1024, 32, 1))
        buf400 = empty_strided_cuda((4, 150, 16, 16), (38400, 256, 16, 1), torch.float32)
        buf401 = empty_strided_cuda((4, 150, 16, 16), (38400, 256, 16, 1), torch.float32)
        buf449 = empty_strided_cuda((4, 246, 16, 16), (62976, 256, 16, 1), torch.float32)
        buf440 = reinterpret_tensor(buf449, (4, 150, 16, 16), (62976, 256, 16, 1), 0)  # alias
        buf464 = empty_strided_cuda((4, 258, 16, 16), (66048, 256, 16, 1), torch.float32)
        buf454 = reinterpret_tensor(buf464, (4, 150, 16, 16), (66048, 256, 16, 1), 0)  # alias
        buf480 = empty_strided_cuda((4, 270, 16, 16), (69120, 256, 16, 1), torch.float32)
        buf469 = reinterpret_tensor(buf480, (4, 150, 16, 16), (69120, 256, 16, 1), 0)  # alias
        buf497 = empty_strided_cuda((4, 282, 16, 16), (72192, 256, 16, 1), torch.float32)
        buf485 = reinterpret_tensor(buf497, (4, 150, 16, 16), (72192, 256, 16, 1), 0)  # alias
        buf515 = empty_strided_cuda((4, 294, 16, 16), (75264, 256, 16, 1), torch.float32)
        buf502 = reinterpret_tensor(buf515, (4, 150, 16, 16), (75264, 256, 16, 1), 0)  # alias
        buf534 = empty_strided_cuda((4, 306, 16, 16), (78336, 256, 16, 1), torch.float32)
        buf520 = reinterpret_tensor(buf534, (4, 150, 16, 16), (78336, 256, 16, 1), 0)  # alias
        buf554 = empty_strided_cuda((4, 318, 16, 16), (81408, 256, 16, 1), torch.float32)
        buf539 = reinterpret_tensor(buf554, (4, 150, 16, 16), (81408, 256, 16, 1), 0)  # alias
        buf575 = empty_strided_cuda((4, 330, 16, 16), (84480, 256, 16, 1), torch.float32)
        buf559 = reinterpret_tensor(buf575, (4, 150, 16, 16), (84480, 256, 16, 1), 0)  # alias
        buf597 = empty_strided_cuda((4, 342, 16, 16), (87552, 256, 16, 1), torch.float32)
        buf580 = reinterpret_tensor(buf597, (4, 150, 16, 16), (87552, 256, 16, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_11, batch_norm_66, relu_66, concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44, concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_56.run(buf399, primals_333, primals_334, primals_335, primals_336, buf400, buf401, buf440, buf454, buf469, buf485, buf502, buf520, buf539, buf559, buf580, 153600, grid=grid(153600), stream=stream0)
        del primals_336
        # Topologically Sorted Source Nodes: [bottleneck_output_32], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_337, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf403 = reinterpret_tensor(buf379, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_67, relu_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf402, primals_338, primals_339, primals_340, primals_341, buf403, 49152, grid=grid(49152), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [new_features_32], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf405 = empty_strided_cuda((4, 162, 16, 16), (41472, 256, 16, 1), torch.float32)
        buf406 = empty_strided_cuda((4, 162, 16, 16), (41472, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_33, batch_norm_68, relu_68], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58.run(buf400, buf404, primals_343, primals_344, primals_345, primals_346, buf405, buf406, 165888, grid=grid(165888), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [bottleneck_output_33], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf408 = reinterpret_tensor(buf301, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_69, relu_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf407, primals_348, primals_349, primals_350, primals_351, buf408, 49152, grid=grid(49152), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [new_features_33], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf410 = empty_strided_cuda((4, 174, 16, 16), (44544, 256, 16, 1), torch.float32)
        buf411 = empty_strided_cuda((4, 174, 16, 16), (44544, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_34, batch_norm_70, relu_70], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59.run(buf400, buf404, buf409, primals_353, primals_354, primals_355, primals_356, buf410, buf411, 178176, grid=grid(178176), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [bottleneck_output_34], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf413 = reinterpret_tensor(buf358, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_71, relu_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf412, primals_358, primals_359, primals_360, primals_361, buf413, 49152, grid=grid(49152), stream=stream0)
        del primals_361
        # Topologically Sorted Source Nodes: [new_features_34], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf415 = empty_strided_cuda((4, 186, 16, 16), (47616, 256, 16, 1), torch.float32)
        buf416 = empty_strided_cuda((4, 186, 16, 16), (47616, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_35, batch_norm_72, relu_72], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60.run(buf400, buf404, buf409, buf414, primals_363, primals_364, primals_365, primals_366, buf415, buf416, 190464, grid=grid(190464), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [bottleneck_output_35], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, primals_367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf418 = reinterpret_tensor(buf284, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_73, relu_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf417, primals_368, primals_369, primals_370, primals_371, buf418, 49152, grid=grid(49152), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [new_features_35], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf420 = empty_strided_cuda((4, 198, 16, 16), (50688, 256, 16, 1), torch.float32)
        buf421 = empty_strided_cuda((4, 198, 16, 16), (50688, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_36, batch_norm_74, relu_74], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61.run(buf400, buf404, buf409, buf414, buf419, primals_373, primals_374, primals_375, primals_376, buf420, buf421, 202752, grid=grid(202752), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [bottleneck_output_36], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_377, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf423 = reinterpret_tensor(buf338, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_75, relu_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf422, primals_378, primals_379, primals_380, primals_381, buf423, 49152, grid=grid(49152), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [new_features_36], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf425 = empty_strided_cuda((4, 210, 16, 16), (53760, 256, 16, 1), torch.float32)
        buf426 = empty_strided_cuda((4, 210, 16, 16), (53760, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_37, batch_norm_76, relu_76], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62.run(buf400, buf404, buf409, buf414, buf419, buf424, primals_383, primals_384, primals_385, primals_386, buf425, buf426, 215040, grid=grid(215040), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [bottleneck_output_37], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf428 = reinterpret_tensor(buf268, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_77, relu_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf427, primals_388, primals_389, primals_390, primals_391, buf428, 49152, grid=grid(49152), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [new_features_37], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf430 = empty_strided_cuda((4, 222, 16, 16), (56832, 256, 16, 1), torch.float32)
        buf431 = empty_strided_cuda((4, 222, 16, 16), (56832, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_38, batch_norm_78, relu_78], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63.run(buf400, buf404, buf409, buf414, buf419, buf424, buf429, primals_393, primals_394, primals_395, primals_396, buf430, buf431, 227328, grid=grid(227328), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [bottleneck_output_38], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf433 = reinterpret_tensor(buf319, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_79, relu_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf432, primals_398, primals_399, primals_400, primals_401, buf433, 49152, grid=grid(49152), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [new_features_38], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf435 = empty_strided_cuda((4, 234, 16, 16), (59904, 256, 16, 1), torch.float32)
        buf436 = empty_strided_cuda((4, 234, 16, 16), (59904, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concated_features_39, batch_norm_80, relu_80], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64.run(buf400, buf404, buf409, buf414, buf419, buf424, buf429, buf434, primals_403, primals_404, primals_405, primals_406, buf435, buf436, 239616, grid=grid(239616), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [bottleneck_output_39], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, primals_407, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf438 = reinterpret_tensor(buf253, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_81, relu_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf437, primals_408, primals_409, primals_410, primals_411, buf438, 49152, grid=grid(49152), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [new_features_39], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf441 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 38400)  # alias
        buf455 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 38400)  # alias
        buf470 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 38400)  # alias
        buf486 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 38400)  # alias
        buf503 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 38400)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf404, buf441, buf455, buf470, buf486, buf503, 12288, grid=grid(12288), stream=stream0)
        buf442 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 41472)  # alias
        buf456 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 41472)  # alias
        buf471 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 41472)  # alias
        buf487 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 41472)  # alias
        buf504 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 41472)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf409, buf442, buf456, buf471, buf487, buf504, 12288, grid=grid(12288), stream=stream0)
        buf443 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 44544)  # alias
        buf457 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 44544)  # alias
        buf472 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 44544)  # alias
        buf488 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 44544)  # alias
        buf505 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 44544)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf414, buf443, buf457, buf472, buf488, buf505, 12288, grid=grid(12288), stream=stream0)
        buf444 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 47616)  # alias
        buf458 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 47616)  # alias
        buf473 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 47616)  # alias
        buf489 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 47616)  # alias
        buf506 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 47616)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf419, buf444, buf458, buf473, buf489, buf506, 12288, grid=grid(12288), stream=stream0)
        buf445 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 50688)  # alias
        buf459 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 50688)  # alias
        buf474 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 50688)  # alias
        buf490 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 50688)  # alias
        buf507 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 50688)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf424, buf445, buf459, buf474, buf490, buf507, 12288, grid=grid(12288), stream=stream0)
        buf446 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 53760)  # alias
        buf460 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 53760)  # alias
        buf475 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 53760)  # alias
        buf491 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 53760)  # alias
        buf508 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 53760)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf429, buf446, buf460, buf475, buf491, buf508, 12288, grid=grid(12288), stream=stream0)
        buf447 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 56832)  # alias
        buf461 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 56832)  # alias
        buf476 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 56832)  # alias
        buf492 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 56832)  # alias
        buf509 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 56832)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf434, buf447, buf461, buf476, buf492, buf509, 12288, grid=grid(12288), stream=stream0)
        buf448 = reinterpret_tensor(buf449, (4, 12, 16, 16), (62976, 256, 16, 1), 59904)  # alias
        buf462 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 59904)  # alias
        buf477 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 59904)  # alias
        buf493 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 59904)  # alias
        buf510 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 59904)  # alias
        # Topologically Sorted Source Nodes: [concated_features_40, concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_65.run(buf439, buf448, buf462, buf477, buf493, buf510, 12288, grid=grid(12288), stream=stream0)
        buf450 = empty_strided_cuda((4, 246, 16, 16), (62976, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_82, relu_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf449, primals_413, primals_414, primals_415, primals_416, buf450, 251904, grid=grid(251904), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [bottleneck_output_40], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, primals_417, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf452 = reinterpret_tensor(buf239, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_83, relu_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf451, primals_418, primals_419, primals_420, primals_421, buf452, 49152, grid=grid(49152), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [new_features_40], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf463 = reinterpret_tensor(buf464, (4, 12, 16, 16), (66048, 256, 16, 1), 62976)  # alias
        buf478 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 62976)  # alias
        buf494 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 62976)  # alias
        buf511 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 62976)  # alias
        # Topologically Sorted Source Nodes: [concated_features_41, concated_features_42, concated_features_43, concated_features_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_67.run(buf453, buf463, buf478, buf494, buf511, 12288, grid=grid(12288), stream=stream0)
        buf465 = empty_strided_cuda((4, 258, 16, 16), (66048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_84, relu_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_68.run(buf464, primals_423, primals_424, primals_425, primals_426, buf465, 264192, grid=grid(264192), stream=stream0)
        del primals_426
        # Topologically Sorted Source Nodes: [bottleneck_output_41], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, primals_427, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf467 = reinterpret_tensor(buf234, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_85, relu_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf466, primals_428, primals_429, primals_430, primals_431, buf467, 49152, grid=grid(49152), stream=stream0)
        del primals_431
        # Topologically Sorted Source Nodes: [new_features_41], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(buf467, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf479 = reinterpret_tensor(buf480, (4, 12, 16, 16), (69120, 256, 16, 1), 66048)  # alias
        buf495 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 66048)  # alias
        buf512 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 66048)  # alias
        buf530 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 66048)  # alias
        # Topologically Sorted Source Nodes: [concated_features_42, concated_features_43, concated_features_44, concated_features_45], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_69.run(buf468, buf479, buf495, buf512, buf530, 12288, grid=grid(12288), stream=stream0)
        buf481 = empty_strided_cuda((4, 270, 16, 16), (69120, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_86, relu_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_70.run(buf480, primals_433, primals_434, primals_435, primals_436, buf481, 276480, grid=grid(276480), stream=stream0)
        del primals_436
        # Topologically Sorted Source Nodes: [bottleneck_output_42], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, primals_437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf483 = reinterpret_tensor(buf229, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_87, relu_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf482, primals_438, primals_439, primals_440, primals_441, buf483, 49152, grid=grid(49152), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [new_features_42], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf496 = reinterpret_tensor(buf497, (4, 12, 16, 16), (72192, 256, 16, 1), 69120)  # alias
        buf513 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 69120)  # alias
        buf531 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 69120)  # alias
        buf550 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 69120)  # alias
        # Topologically Sorted Source Nodes: [concated_features_43, concated_features_44, concated_features_45, concated_features_46], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_71.run(buf484, buf496, buf513, buf531, buf550, 12288, grid=grid(12288), stream=stream0)
        buf498 = empty_strided_cuda((4, 282, 16, 16), (72192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_88, relu_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf497, primals_443, primals_444, primals_445, primals_446, buf498, 288768, grid=grid(288768), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [bottleneck_output_43], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf500 = reinterpret_tensor(buf224, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_89, relu_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf499, primals_448, primals_449, primals_450, primals_451, buf500, 49152, grid=grid(49152), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [new_features_43], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_452, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf514 = reinterpret_tensor(buf515, (4, 12, 16, 16), (75264, 256, 16, 1), 72192)  # alias
        buf532 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 72192)  # alias
        buf551 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 72192)  # alias
        buf571 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 72192)  # alias
        # Topologically Sorted Source Nodes: [concated_features_44, concated_features_45, concated_features_46, concated_features_47], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_73.run(buf501, buf514, buf532, buf551, buf571, 12288, grid=grid(12288), stream=stream0)
        buf516 = empty_strided_cuda((4, 294, 16, 16), (75264, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_90, relu_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_74.run(buf515, primals_453, primals_454, primals_455, primals_456, buf516, 301056, grid=grid(301056), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [bottleneck_output_44], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_457, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf518 = reinterpret_tensor(buf219, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_91, relu_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf517, primals_458, primals_459, primals_460, primals_461, buf518, 49152, grid=grid(49152), stream=stream0)
        del primals_461
        # Topologically Sorted Source Nodes: [new_features_44], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf518, primals_462, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf521 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 38400)  # alias
        buf540 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 38400)  # alias
        buf560 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 38400)  # alias
        buf581 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 38400)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf404, buf521, buf540, buf560, buf581, 12288, grid=grid(12288), stream=stream0)
        del buf404
        buf522 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 41472)  # alias
        buf541 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 41472)  # alias
        buf561 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 41472)  # alias
        buf582 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 41472)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf409, buf522, buf541, buf561, buf582, 12288, grid=grid(12288), stream=stream0)
        del buf409
        buf523 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 44544)  # alias
        buf542 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 44544)  # alias
        buf562 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 44544)  # alias
        buf583 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 44544)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf414, buf523, buf542, buf562, buf583, 12288, grid=grid(12288), stream=stream0)
        del buf414
        buf524 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 47616)  # alias
        buf543 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 47616)  # alias
        buf563 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 47616)  # alias
        buf584 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 47616)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf419, buf524, buf543, buf563, buf584, 12288, grid=grid(12288), stream=stream0)
        del buf419
        buf525 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 50688)  # alias
        buf544 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 50688)  # alias
        buf564 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 50688)  # alias
        buf585 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 50688)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf424, buf525, buf544, buf564, buf585, 12288, grid=grid(12288), stream=stream0)
        del buf424
        buf526 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 53760)  # alias
        buf545 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 53760)  # alias
        buf565 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 53760)  # alias
        buf586 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 53760)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf429, buf526, buf545, buf565, buf586, 12288, grid=grid(12288), stream=stream0)
        del buf429
        buf527 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 56832)  # alias
        buf546 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 56832)  # alias
        buf566 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 56832)  # alias
        buf587 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 56832)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf434, buf527, buf546, buf566, buf587, 12288, grid=grid(12288), stream=stream0)
        del buf434
        buf528 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 59904)  # alias
        buf547 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 59904)  # alias
        buf567 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 59904)  # alias
        buf588 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 59904)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf439, buf528, buf547, buf567, buf588, 12288, grid=grid(12288), stream=stream0)
        del buf439
        buf529 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 62976)  # alias
        buf548 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 62976)  # alias
        buf568 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 62976)  # alias
        buf589 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 62976)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf453, buf529, buf548, buf568, buf589, 12288, grid=grid(12288), stream=stream0)
        del buf453
        buf533 = reinterpret_tensor(buf534, (4, 12, 16, 16), (78336, 256, 16, 1), 75264)  # alias
        buf552 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 75264)  # alias
        buf572 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 75264)  # alias
        buf593 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [concated_features_45, concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf519, buf533, buf552, buf572, buf593, 12288, grid=grid(12288), stream=stream0)
        del buf519
        buf535 = empty_strided_cuda((4, 306, 16, 16), (78336, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_92, relu_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_76.run(buf534, primals_463, primals_464, primals_465, primals_466, buf535, 313344, grid=grid(313344), stream=stream0)
        del primals_466
        # Topologically Sorted Source Nodes: [bottleneck_output_45], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf537 = reinterpret_tensor(buf214, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_93, relu_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf536, primals_468, primals_469, primals_470, primals_471, buf537, 49152, grid=grid(49152), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [new_features_45], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, primals_472, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf549 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 66048)  # alias
        buf569 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 66048)  # alias
        buf590 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 66048)  # alias
        # Topologically Sorted Source Nodes: [concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_77.run(buf468, buf549, buf569, buf590, 12288, grid=grid(12288), stream=stream0)
        del buf468
        buf553 = reinterpret_tensor(buf554, (4, 12, 16, 16), (81408, 256, 16, 1), 78336)  # alias
        buf573 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 78336)  # alias
        buf594 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 78336)  # alias
        # Topologically Sorted Source Nodes: [concated_features_46, concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_77.run(buf538, buf553, buf573, buf594, 12288, grid=grid(12288), stream=stream0)
        del buf538
        buf555 = empty_strided_cuda((4, 318, 16, 16), (81408, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_94, relu_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_78.run(buf554, primals_473, primals_474, primals_475, primals_476, buf555, 325632, grid=grid(325632), stream=stream0)
        del primals_476
        # Topologically Sorted Source Nodes: [bottleneck_output_46], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(buf555, primals_477, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf557 = reinterpret_tensor(buf209, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_95, relu_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf556, primals_478, primals_479, primals_480, primals_481, buf557, 49152, grid=grid(49152), stream=stream0)
        del primals_481
        # Topologically Sorted Source Nodes: [new_features_46], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, primals_482, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf570 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 69120)  # alias
        buf591 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 69120)  # alias
        # Topologically Sorted Source Nodes: [concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf484, buf570, buf591, 12288, grid=grid(12288), stream=stream0)
        del buf484
        buf574 = reinterpret_tensor(buf575, (4, 12, 16, 16), (84480, 256, 16, 1), 81408)  # alias
        buf595 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 81408)  # alias
        # Topologically Sorted Source Nodes: [concated_features_47, input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_79.run(buf558, buf574, buf595, 12288, grid=grid(12288), stream=stream0)
        del buf558
        buf576 = empty_strided_cuda((4, 330, 16, 16), (84480, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_96, relu_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_80.run(buf575, primals_483, primals_484, primals_485, primals_486, buf576, 337920, grid=grid(337920), stream=stream0)
        del primals_486
        # Topologically Sorted Source Nodes: [bottleneck_output_47], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(buf576, primals_487, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf577, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf578 = reinterpret_tensor(buf204, (4, 48, 16, 16), (12288, 256, 16, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_97, relu_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_57.run(buf577, primals_488, primals_489, primals_490, primals_491, buf578, 49152, grid=grid(49152), stream=stream0)
        del primals_491
        # Topologically Sorted Source Nodes: [new_features_47], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, primals_492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf592 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 72192)  # alias
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf501, buf592, 12288, grid=grid(12288), stream=stream0)
        del buf501
        buf596 = reinterpret_tensor(buf597, (4, 12, 16, 16), (87552, 256, 16, 1), 84480)  # alias
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_81.run(buf579, buf596, 12288, grid=grid(12288), stream=stream0)
        del buf579
        buf598 = empty_strided_cuda((4, 342, 1, 1), (342, 1, 1368, 1368), torch.float32)
        buf599 = buf598; del buf598  # reuse
        # Topologically Sorted Source Nodes: [input_13, out, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_82.run(buf599, buf597, primals_493, primals_494, primals_495, primals_496, 1368, 256, grid=grid(1368), stream=stream0)
        buf600 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_498, reinterpret_tensor(buf599, (4, 342), (342, 1), 0), reinterpret_tensor(primals_497, (342, 10), (1, 342), 0), alpha=1, beta=1, out=buf600)
        del primals_498
    return (buf600, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, primals_239, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_307, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_315, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_342, primals_343, primals_344, primals_345, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_357, primals_358, primals_359, primals_360, primals_362, primals_363, primals_364, primals_365, primals_367, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_385, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_402, primals_403, primals_404, primals_405, primals_407, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_415, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, primals_462, primals_463, primals_464, primals_465, primals_467, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_475, primals_477, primals_478, primals_479, primals_480, primals_482, primals_483, primals_484, primals_485, primals_487, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_495, primals_496, buf0, buf1, buf2, buf3, buf5, buf6, buf7, buf8, buf10, buf11, buf12, buf13, buf15, buf16, buf17, buf18, buf20, buf21, buf22, buf23, buf25, buf26, buf27, buf28, buf30, buf31, buf32, buf33, buf35, buf36, buf37, buf38, buf49, buf50, buf51, buf52, buf64, buf65, buf66, buf67, buf80, buf81, buf82, buf83, buf97, buf98, buf99, buf100, buf115, buf116, buf117, buf118, buf134, buf135, buf136, buf137, buf154, buf155, buf156, buf157, buf175, buf176, buf177, buf178, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf205, buf206, buf207, buf208, buf210, buf211, buf212, buf213, buf215, buf216, buf217, buf218, buf220, buf221, buf222, buf223, buf225, buf226, buf227, buf228, buf230, buf231, buf232, buf233, buf235, buf236, buf237, buf238, buf249, buf250, buf251, buf252, buf264, buf265, buf266, buf267, buf280, buf281, buf282, buf283, buf297, buf298, buf299, buf300, buf315, buf316, buf317, buf318, buf334, buf335, buf336, buf337, buf354, buf355, buf356, buf357, buf375, buf376, buf377, buf378, buf397, buf398, buf399, buf400, buf401, buf402, buf403, buf405, buf406, buf407, buf408, buf410, buf411, buf412, buf413, buf415, buf416, buf417, buf418, buf420, buf421, buf422, buf423, buf425, buf426, buf427, buf428, buf430, buf431, buf432, buf433, buf435, buf436, buf437, buf438, buf449, buf450, buf451, buf452, buf464, buf465, buf466, buf467, buf480, buf481, buf482, buf483, buf497, buf498, buf499, buf500, buf515, buf516, buf517, buf518, buf534, buf535, buf536, buf537, buf554, buf555, buf556, buf557, buf575, buf576, buf577, buf578, buf597, reinterpret_tensor(buf599, (4, 342), (342, 1), 0), primals_497, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((48, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((48, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((48, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((48, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((48, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((48, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((48, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((48, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((48, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((48, 156, 1, 1), (156, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((48, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((48, 180, 1, 1), (180, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((48, 204, 1, 1), (204, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((108, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((48, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((48, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((48, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((48, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((156, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((48, 156, 1, 1), (156, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((48, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((180, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((48, 180, 1, 1), (180, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((204, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((48, 204, 1, 1), (204, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((48, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((48, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((48, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((252, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((252, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((252, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((252, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((48, 252, 1, 1), (252, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((48, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((48, 276, 1, 1), (276, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((150, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((48, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((48, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((48, 174, 1, 1), (174, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((186, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((186, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((186, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((186, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((48, 186, 1, 1), (186, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((198, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((198, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((198, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((198, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((48, 198, 1, 1), (198, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((210, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((210, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((210, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((210, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((48, 210, 1, 1), (210, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((222, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((222, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((222, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((222, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((48, 222, 1, 1), (222, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((234, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((234, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((234, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((234, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((48, 234, 1, 1), (234, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((246, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((246, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((246, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((246, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((48, 246, 1, 1), (246, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((258, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((258, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((258, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((258, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((48, 258, 1, 1), (258, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((270, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((270, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((270, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((270, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((48, 270, 1, 1), (270, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((48, 282, 1, 1), (282, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((294, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((294, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((294, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((294, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((48, 294, 1, 1), (294, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((306, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((306, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((306, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((306, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((48, 306, 1, 1), (306, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((318, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((318, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((318, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((318, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((48, 318, 1, 1), (318, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((330, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((330, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((330, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((330, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((48, 330, 1, 1), (330, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((342, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((342, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((342, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((342, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((10, 342), (342, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
