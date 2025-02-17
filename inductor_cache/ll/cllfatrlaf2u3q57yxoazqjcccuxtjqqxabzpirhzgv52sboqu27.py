# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/dn/cdnwnes55tpuybuqyysg7oijk7437v44vutbbkfx7g43sqpodjuu.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_2, %slice_4, %slice_6, %slice_8], 1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 4) % 12)
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x3 = xindex // 48
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x0 + 8*x1 + 16*(x2) + 48*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 6, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1 + 16*((-3) + x2) + 48*x3), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 9, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1 + 16*((-6) + x2) + 48*x3), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 12, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1 + 16*((-9) + x2) + 48*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x4), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/57/c572rlapdimw74hgq23frypo35cpuuqc4k45fchtsrjzbznt4zcz.py
# Topologically Sorted Source Nodes: [batch_norm, sigmoid, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   sigmoid => sigmoid
#   x => mul_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5v/c5vsh62jgabwu7mxztrbjshzhv2bsgem6dx47jrwt5u2qrjiscyj.py
# Topologically Sorted Source Nodes: [batch_norm_1, sigmoid_1, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_5, mul_6, sub_1
#   input_1 => mul_7
#   sigmoid_1 => sigmoid_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_7 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3l/c3lmz7o2l536qhnrvy7a6uxskmuire7fe47mdhkok5zp7c3jbfto.py
# Topologically Sorted Source Nodes: [batch_norm_2, sigmoid_2, mul_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_2 => add_5, mul_10, mul_9, sub_2
#   mul_2 => mul_11
#   sigmoid_2 => sigmoid_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_5,), kwargs = {})
#   %mul_11 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %sigmoid_2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zz/czzvajepn6d2gteagtdnuxd3tygdgri2tbjkgqdposcdp2c4wryn.py
# Topologically Sorted Source Nodes: [batch_norm_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_4 => add_9, mul_17, mul_18, sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/22/c22e7oqmbugysyaen5kuovf6ulfktb2kugfuzx45cvxjhrww6j4o.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_10, %mul_23], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 8, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (4*x1 + ((-4) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp11, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cl/cclogah3iupiysacjr3v3g4xjawb3w75vvvh55y4zeilcrmqyqkk.py
# Topologically Sorted Source Nodes: [batch_norm_7, sigmoid_7, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_7 => add_16, mul_29, mul_30, sub_7
#   input_4 => mul_31
#   sigmoid_7 => sigmoid_7
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_61), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %unsqueeze_63), kwargs = {})
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_16,), kwargs = {})
#   %mul_31 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sigmoid_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pl/cplgo2tcocgowt77kqeklk5ulbtksm4u54zkfcayg4nbrqncr3cg.py
# Topologically Sorted Source Nodes: [batch_norm_10, sigmoid_10, mul_10, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   batch_norm_10 => add_22, mul_41, mul_42, sub_10
#   input_5 => add_23
#   mul_10 => mul_43
#   sigmoid_10 => sigmoid_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_85), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %unsqueeze_87), kwargs = {})
#   %sigmoid_10 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_10), kwargs = {})
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %mul_43), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tl.sigmoid(tmp15)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp16 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ij/cijfxlienlmtvlxxmf5ingflt4usy53pgbb22ch2frw4pbtaw5x2.py
# Topologically Sorted Source Nodes: [batch_norm_14], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_14 => add_32, mul_57, mul_58, sub_14
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %unsqueeze_117), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_58, %unsqueeze_119), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ae/caezzytxhmhtg3bsrh7zelyzftmgnvgc47fpo2e42genqqs62em4.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_33, %mul_63], 1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 16, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (8*x1 + ((-8) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp11, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gl/cgldzt7f34ut2yvnf5q5b3oetrnd3qyuaci6pbalr4l3extsvo2r.py
# Topologically Sorted Source Nodes: [batch_norm_17, sigmoid_17, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_17 => add_39, mul_69, mul_70, sub_17
#   input_9 => mul_71
#   sigmoid_17 => sigmoid_17
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %unsqueeze_141), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %unsqueeze_143), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_39,), kwargs = {})
#   %mul_71 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_39, %sigmoid_17), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4e/c4eylzxte44hqprxqzpmm3lulxkyic3ta6wjnvelxksopa7k6fao.py
# Topologically Sorted Source Nodes: [batch_norm_20, sigmoid_20, mul_20, input_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   batch_norm_20 => add_45, mul_81, mul_82, sub_20
#   input_10 => add_46
#   mul_20 => mul_83
#   sigmoid_20 => sigmoid_20
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %unsqueeze_165), kwargs = {})
#   %add_45 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %unsqueeze_167), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_45,), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %sigmoid_20), kwargs = {})
#   %add_46 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %mul_83), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tl.sigmoid(tmp15)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp16 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n3/cn3po37cabzismb62oyi4kwtchnkbwnvedm4b25wa6bqdzz4zwj4.py
# Topologically Sorted Source Nodes: [batch_norm_24], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_24 => add_55, mul_97, mul_98, sub_24
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_197), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_199), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fy/cfy77gwj6yduaoqtbuyw4mj6oqxvwhdkp4mkbtf7xxc5p2anwmid.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_56, %mul_103], 1), kwargs = {})
triton_poi_fused_cat_13 = async_compile.triton('triton_poi_fused_cat_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (16*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 32, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (16*x1 + ((-16) + x0)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp11, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ox/cox5jwl4s56vi3sn4fce7l35rmnfn6imcund3owouoltxyrovqx4.py
# Topologically Sorted Source Nodes: [batch_norm_27, sigmoid_27, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   batch_norm_27 => add_62, mul_109, mul_110, sub_27
#   input_14 => mul_111
#   sigmoid_27 => sigmoid_27
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_221), kwargs = {})
#   %add_62 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_223), kwargs = {})
#   %sigmoid_27 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_62,), kwargs = {})
#   %mul_111 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_62, %sigmoid_27), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d2/cd27ih6dcelhi4vhwezzf2gkey4fg5vjphhs77dy2b2z7ufbpuqw.py
# Topologically Sorted Source Nodes: [batch_norm_28, sigmoid_28, x_1, max_pool2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   batch_norm_28 => add_64, mul_113, mul_114, sub_28
#   max_pool2d => getitem_1
#   sigmoid_28 => sigmoid_28
#   x_1 => mul_115
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_229), kwargs = {})
#   %add_64 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_231), kwargs = {})
#   %sigmoid_28 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_64,), kwargs = {})
#   %mul_115 : [num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, %sigmoid_28), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_mul_sigmoid_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_mul_sigmoid_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_mul_sigmoid_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_mul_sigmoid_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1], -2, tl.int64)
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.full([1], 1, tl.int64)
    tmp22 = tmp18 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tmp23 & tmp23
    tmp25 = tl.full([1], -1, tl.int64)
    tmp26 = tmp25 >= tmp19
    tmp27 = tmp25 < tmp21
    tmp28 = tmp26 & tmp27
    tmp29 = tmp23 & tmp28
    tmp30 = tmp17 > tmp17
    tmp31 = tl.full([1], 1, tl.int8)
    tmp32 = tl.full([1], 0, tl.int8)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = triton_helpers.maximum(tmp17, tmp17)
    tmp35 = tmp19 >= tmp19
    tmp36 = tmp19 < tmp21
    tmp37 = tmp35 & tmp36
    tmp38 = tmp23 & tmp37
    tmp39 = tmp17 > tmp34
    tmp40 = tl.full([1], 2, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp33)
    tmp42 = triton_helpers.maximum(tmp17, tmp34)
    tmp43 = tmp21 >= tmp19
    tmp44 = tmp21 < tmp21
    tmp45 = tmp43 & tmp44
    tmp46 = tmp23 & tmp45
    tmp47 = tmp17 > tmp42
    tmp48 = tl.full([1], 3, tl.int8)
    tmp49 = tl.where(tmp47, tmp48, tmp41)
    tmp50 = triton_helpers.maximum(tmp17, tmp42)
    tmp51 = tl.full([1], 2, tl.int64)
    tmp52 = tmp51 >= tmp19
    tmp53 = tmp51 < tmp21
    tmp54 = tmp52 & tmp53
    tmp55 = tmp23 & tmp54
    tmp56 = tmp17 > tmp50
    tmp57 = tl.full([1], 4, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp49)
    tmp59 = triton_helpers.maximum(tmp17, tmp50)
    tmp60 = tmp28 & tmp23
    tmp61 = tmp17 > tmp59
    tmp62 = tl.full([1], 5, tl.int8)
    tmp63 = tl.where(tmp61, tmp62, tmp58)
    tmp64 = triton_helpers.maximum(tmp17, tmp59)
    tmp65 = tmp28 & tmp28
    tmp66 = tmp17 > tmp64
    tmp67 = tl.full([1], 6, tl.int8)
    tmp68 = tl.where(tmp66, tmp67, tmp63)
    tmp69 = triton_helpers.maximum(tmp17, tmp64)
    tmp70 = tmp28 & tmp37
    tmp71 = tmp17 > tmp69
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp68)
    tmp74 = triton_helpers.maximum(tmp17, tmp69)
    tmp75 = tmp28 & tmp45
    tmp76 = tmp17 > tmp74
    tmp77 = tl.full([1], 8, tl.int8)
    tmp78 = tl.where(tmp76, tmp77, tmp73)
    tmp79 = triton_helpers.maximum(tmp17, tmp74)
    tmp80 = tmp28 & tmp54
    tmp81 = tmp17 > tmp79
    tmp82 = tl.full([1], 9, tl.int8)
    tmp83 = tl.where(tmp81, tmp82, tmp78)
    tmp84 = triton_helpers.maximum(tmp17, tmp79)
    tmp85 = tmp37 & tmp23
    tmp86 = tmp17 > tmp84
    tmp87 = tl.full([1], 10, tl.int8)
    tmp88 = tl.where(tmp86, tmp87, tmp83)
    tmp89 = triton_helpers.maximum(tmp17, tmp84)
    tmp90 = tmp37 & tmp28
    tmp91 = tmp17 > tmp89
    tmp92 = tl.full([1], 11, tl.int8)
    tmp93 = tl.where(tmp91, tmp92, tmp88)
    tmp94 = triton_helpers.maximum(tmp17, tmp89)
    tmp95 = tmp37 & tmp37
    tmp96 = tmp17 > tmp94
    tmp97 = tl.full([1], 12, tl.int8)
    tmp98 = tl.where(tmp96, tmp97, tmp93)
    tmp99 = triton_helpers.maximum(tmp17, tmp94)
    tmp100 = tmp37 & tmp45
    tmp101 = tmp17 > tmp99
    tmp102 = tl.full([1], 13, tl.int8)
    tmp103 = tl.where(tmp101, tmp102, tmp98)
    tmp104 = triton_helpers.maximum(tmp17, tmp99)
    tmp105 = tmp37 & tmp54
    tmp106 = tmp17 > tmp104
    tmp107 = tl.full([1], 14, tl.int8)
    tmp108 = tl.where(tmp106, tmp107, tmp103)
    tmp109 = triton_helpers.maximum(tmp17, tmp104)
    tmp110 = tmp45 & tmp23
    tmp111 = tmp17 > tmp109
    tmp112 = tl.full([1], 15, tl.int8)
    tmp113 = tl.where(tmp111, tmp112, tmp108)
    tmp114 = triton_helpers.maximum(tmp17, tmp109)
    tmp115 = tmp45 & tmp28
    tmp116 = tmp17 > tmp114
    tmp117 = tl.full([1], 16, tl.int8)
    tmp118 = tl.where(tmp116, tmp117, tmp113)
    tmp119 = triton_helpers.maximum(tmp17, tmp114)
    tmp120 = tmp45 & tmp37
    tmp121 = tmp17 > tmp119
    tmp122 = tl.full([1], 17, tl.int8)
    tmp123 = tl.where(tmp121, tmp122, tmp118)
    tmp124 = triton_helpers.maximum(tmp17, tmp119)
    tmp125 = tmp45 & tmp45
    tmp126 = tmp17 > tmp124
    tmp127 = tl.full([1], 18, tl.int8)
    tmp128 = tl.where(tmp126, tmp127, tmp123)
    tmp129 = triton_helpers.maximum(tmp17, tmp124)
    tmp130 = tmp45 & tmp54
    tmp131 = tmp17 > tmp129
    tmp132 = tl.full([1], 19, tl.int8)
    tmp133 = tl.where(tmp131, tmp132, tmp128)
    tmp134 = triton_helpers.maximum(tmp17, tmp129)
    tmp135 = tmp54 & tmp23
    tmp136 = tmp17 > tmp134
    tmp137 = tl.full([1], 20, tl.int8)
    tmp138 = tl.where(tmp136, tmp137, tmp133)
    tmp139 = triton_helpers.maximum(tmp17, tmp134)
    tmp140 = tmp54 & tmp28
    tmp141 = tmp17 > tmp139
    tmp142 = tl.full([1], 21, tl.int8)
    tmp143 = tl.where(tmp141, tmp142, tmp138)
    tmp144 = triton_helpers.maximum(tmp17, tmp139)
    tmp145 = tmp54 & tmp37
    tmp146 = tmp17 > tmp144
    tmp147 = tl.full([1], 22, tl.int8)
    tmp148 = tl.where(tmp146, tmp147, tmp143)
    tmp149 = triton_helpers.maximum(tmp17, tmp144)
    tmp150 = tmp54 & tmp45
    tmp151 = tmp17 > tmp149
    tmp152 = tl.full([1], 23, tl.int8)
    tmp153 = tl.where(tmp151, tmp152, tmp148)
    tmp154 = triton_helpers.maximum(tmp17, tmp149)
    tmp155 = tmp54 & tmp54
    tmp156 = tmp17 > tmp154
    tmp157 = tl.full([1], 24, tl.int8)
    tmp158 = tl.where(tmp156, tmp157, tmp153)
    tmp159 = triton_helpers.maximum(tmp17, tmp154)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    tl.store(out_ptr0 + (x2), tmp158, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uj/cujpt5p2unrnzksmzwdpuqzrjntuw2gqobdkigsjd35p642eohyp.py
# Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_4 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_115, %getitem, %getitem_2, %getitem_4], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.full([1], -2, tl.int64)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 1, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp15
    tmp17 = tmp16 & tmp9
    tmp18 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp17 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp19 = tl.full([1], -1, tl.int64)
    tmp20 = tmp19 >= tmp11
    tmp21 = tmp19 < tmp13
    tmp22 = tmp20 & tmp21
    tmp23 = tmp15 & tmp22
    tmp24 = tmp23 & tmp9
    tmp25 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp26 = triton_helpers.maximum(tmp25, tmp18)
    tmp27 = tmp11 >= tmp11
    tmp28 = tmp11 < tmp13
    tmp29 = tmp27 & tmp28
    tmp30 = tmp15 & tmp29
    tmp31 = tmp30 & tmp9
    tmp32 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp31 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp26)
    tmp34 = tmp13 >= tmp11
    tmp35 = tmp13 < tmp13
    tmp36 = tmp34 & tmp35
    tmp37 = tmp15 & tmp36
    tmp38 = tmp37 & tmp9
    tmp39 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp38 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp33)
    tmp41 = tl.full([1], 2, tl.int64)
    tmp42 = tmp41 >= tmp11
    tmp43 = tmp41 < tmp13
    tmp44 = tmp42 & tmp43
    tmp45 = tmp15 & tmp44
    tmp46 = tmp45 & tmp9
    tmp47 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp40)
    tmp49 = tmp22 & tmp15
    tmp50 = tmp49 & tmp9
    tmp51 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp50 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp48)
    tmp53 = tmp22 & tmp22
    tmp54 = tmp53 & tmp9
    tmp55 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp54 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp52)
    tmp57 = tmp22 & tmp29
    tmp58 = tmp57 & tmp9
    tmp59 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp58 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp56)
    tmp61 = tmp22 & tmp36
    tmp62 = tmp61 & tmp9
    tmp63 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp62 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp64 = triton_helpers.maximum(tmp63, tmp60)
    tmp65 = tmp22 & tmp44
    tmp66 = tmp65 & tmp9
    tmp67 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp66 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp68 = triton_helpers.maximum(tmp67, tmp64)
    tmp69 = tmp29 & tmp15
    tmp70 = tmp69 & tmp9
    tmp71 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp70 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp72 = triton_helpers.maximum(tmp71, tmp68)
    tmp73 = tmp29 & tmp22
    tmp74 = tmp73 & tmp9
    tmp75 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp74 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp76 = triton_helpers.maximum(tmp75, tmp72)
    tmp77 = tmp29 & tmp29
    tmp78 = tmp77 & tmp9
    tmp79 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp78 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp80 = triton_helpers.maximum(tmp79, tmp76)
    tmp81 = tmp29 & tmp36
    tmp82 = tmp81 & tmp9
    tmp83 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp82 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp80)
    tmp85 = tmp29 & tmp44
    tmp86 = tmp85 & tmp9
    tmp87 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp86 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp88 = triton_helpers.maximum(tmp87, tmp84)
    tmp89 = tmp36 & tmp15
    tmp90 = tmp89 & tmp9
    tmp91 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp90 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp92 = triton_helpers.maximum(tmp91, tmp88)
    tmp93 = tmp36 & tmp22
    tmp94 = tmp93 & tmp9
    tmp95 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp94 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp92)
    tmp97 = tmp36 & tmp29
    tmp98 = tmp97 & tmp9
    tmp99 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp98 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp100 = triton_helpers.maximum(tmp99, tmp96)
    tmp101 = tmp36 & tmp36
    tmp102 = tmp101 & tmp9
    tmp103 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp102 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp104 = triton_helpers.maximum(tmp103, tmp100)
    tmp105 = tmp36 & tmp44
    tmp106 = tmp105 & tmp9
    tmp107 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp106 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp108 = triton_helpers.maximum(tmp107, tmp104)
    tmp109 = tmp44 & tmp15
    tmp110 = tmp109 & tmp9
    tmp111 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp110 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp112 = triton_helpers.maximum(tmp111, tmp108)
    tmp113 = tmp44 & tmp22
    tmp114 = tmp113 & tmp9
    tmp115 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp114 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp116 = triton_helpers.maximum(tmp115, tmp112)
    tmp117 = tmp44 & tmp29
    tmp118 = tmp117 & tmp9
    tmp119 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp118 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp120 = triton_helpers.maximum(tmp119, tmp116)
    tmp121 = tmp44 & tmp36
    tmp122 = tmp121 & tmp9
    tmp123 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp122 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp124 = triton_helpers.maximum(tmp123, tmp120)
    tmp125 = tmp44 & tmp44
    tmp126 = tmp125 & tmp9
    tmp127 = tl.load(in_ptr0 + (32*x1 + ((-32) + x0)), tmp126 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp128 = triton_helpers.maximum(tmp127, tmp124)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp9, tmp128, tmp129)
    tmp131 = tmp0 >= tmp7
    tmp132 = tl.full([1], 96, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), tmp134 & xmask, eviction_policy='evict_last', other=0.0)
    tmp136 = tmp0 >= tmp132
    tmp137 = tl.full([1], 128, tl.int64)
    tmp138 = tmp0 < tmp137
    tmp139 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), tmp136 & xmask, eviction_policy='evict_last', other=0.0)
    tmp140 = tl.where(tmp134, tmp135, tmp139)
    tmp141 = tl.where(tmp9, tmp130, tmp140)
    tmp142 = tl.where(tmp4, tmp5, tmp141)
    tl.store(out_ptr0 + (x2), tmp142, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fu/cfube7vkbcjkxpdfcgz7zhilnacdv6edihofwobvzigegokid44j.py
# Topologically Sorted Source Nodes: [batch_norm_32], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_32 => add_72, mul_129, mul_130, sub_32
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_129, %unsqueeze_261), kwargs = {})
#   %add_72 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_130, %unsqueeze_263), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f2/cf2gf7o4sfatsf7qdqp7tb5mujvh7llmsmsi54bkmqawa4ke3wnz.py
# Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_5 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_131, %mul_135], 1), kwargs = {})
triton_poi_fused_cat_18 = async_compile.triton('triton_poi_fused_cat_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 64, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (32*x1 + ((-32) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (4, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (8, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (8, ), (1, ))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, ), (1, ))
    assert_size_stride(primals_12, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_33, (8, ), (1, ))
    assert_size_stride(primals_34, (8, ), (1, ))
    assert_size_stride(primals_35, (8, ), (1, ))
    assert_size_stride(primals_36, (8, ), (1, ))
    assert_size_stride(primals_37, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_38, (16, ), (1, ))
    assert_size_stride(primals_39, (16, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (16, ), (1, ))
    assert_size_stride(primals_42, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_43, (8, ), (1, ))
    assert_size_stride(primals_44, (8, ), (1, ))
    assert_size_stride(primals_45, (8, ), (1, ))
    assert_size_stride(primals_46, (8, ), (1, ))
    assert_size_stride(primals_47, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_48, (8, ), (1, ))
    assert_size_stride(primals_49, (8, ), (1, ))
    assert_size_stride(primals_50, (8, ), (1, ))
    assert_size_stride(primals_51, (8, ), (1, ))
    assert_size_stride(primals_52, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_53, (8, ), (1, ))
    assert_size_stride(primals_54, (8, ), (1, ))
    assert_size_stride(primals_55, (8, ), (1, ))
    assert_size_stride(primals_56, (8, ), (1, ))
    assert_size_stride(primals_57, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_58, (8, ), (1, ))
    assert_size_stride(primals_59, (8, ), (1, ))
    assert_size_stride(primals_60, (8, ), (1, ))
    assert_size_stride(primals_61, (8, ), (1, ))
    assert_size_stride(primals_62, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_63, (8, ), (1, ))
    assert_size_stride(primals_64, (8, ), (1, ))
    assert_size_stride(primals_65, (8, ), (1, ))
    assert_size_stride(primals_66, (8, ), (1, ))
    assert_size_stride(primals_67, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_68, (8, ), (1, ))
    assert_size_stride(primals_69, (8, ), (1, ))
    assert_size_stride(primals_70, (8, ), (1, ))
    assert_size_stride(primals_71, (8, ), (1, ))
    assert_size_stride(primals_72, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_73, (8, ), (1, ))
    assert_size_stride(primals_74, (8, ), (1, ))
    assert_size_stride(primals_75, (8, ), (1, ))
    assert_size_stride(primals_76, (8, ), (1, ))
    assert_size_stride(primals_77, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_78, (8, ), (1, ))
    assert_size_stride(primals_79, (8, ), (1, ))
    assert_size_stride(primals_80, (8, ), (1, ))
    assert_size_stride(primals_81, (8, ), (1, ))
    assert_size_stride(primals_82, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_83, (16, ), (1, ))
    assert_size_stride(primals_84, (16, ), (1, ))
    assert_size_stride(primals_85, (16, ), (1, ))
    assert_size_stride(primals_86, (16, ), (1, ))
    assert_size_stride(primals_87, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_89, (32, ), (1, ))
    assert_size_stride(primals_90, (32, ), (1, ))
    assert_size_stride(primals_91, (32, ), (1, ))
    assert_size_stride(primals_92, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_93, (16, ), (1, ))
    assert_size_stride(primals_94, (16, ), (1, ))
    assert_size_stride(primals_95, (16, ), (1, ))
    assert_size_stride(primals_96, (16, ), (1, ))
    assert_size_stride(primals_97, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_98, (16, ), (1, ))
    assert_size_stride(primals_99, (16, ), (1, ))
    assert_size_stride(primals_100, (16, ), (1, ))
    assert_size_stride(primals_101, (16, ), (1, ))
    assert_size_stride(primals_102, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_103, (16, ), (1, ))
    assert_size_stride(primals_104, (16, ), (1, ))
    assert_size_stride(primals_105, (16, ), (1, ))
    assert_size_stride(primals_106, (16, ), (1, ))
    assert_size_stride(primals_107, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_108, (16, ), (1, ))
    assert_size_stride(primals_109, (16, ), (1, ))
    assert_size_stride(primals_110, (16, ), (1, ))
    assert_size_stride(primals_111, (16, ), (1, ))
    assert_size_stride(primals_112, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_113, (16, ), (1, ))
    assert_size_stride(primals_114, (16, ), (1, ))
    assert_size_stride(primals_115, (16, ), (1, ))
    assert_size_stride(primals_116, (16, ), (1, ))
    assert_size_stride(primals_117, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_118, (16, ), (1, ))
    assert_size_stride(primals_119, (16, ), (1, ))
    assert_size_stride(primals_120, (16, ), (1, ))
    assert_size_stride(primals_121, (16, ), (1, ))
    assert_size_stride(primals_122, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_123, (16, ), (1, ))
    assert_size_stride(primals_124, (16, ), (1, ))
    assert_size_stride(primals_125, (16, ), (1, ))
    assert_size_stride(primals_126, (16, ), (1, ))
    assert_size_stride(primals_127, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_128, (16, ), (1, ))
    assert_size_stride(primals_129, (16, ), (1, ))
    assert_size_stride(primals_130, (16, ), (1, ))
    assert_size_stride(primals_131, (16, ), (1, ))
    assert_size_stride(primals_132, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_133, (32, ), (1, ))
    assert_size_stride(primals_134, (32, ), (1, ))
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, ), (1, ))
    assert_size_stride(primals_137, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, ), (1, ))
    assert_size_stride(primals_141, (64, ), (1, ))
    assert_size_stride(primals_142, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_143, (32, ), (1, ))
    assert_size_stride(primals_144, (32, ), (1, ))
    assert_size_stride(primals_145, (32, ), (1, ))
    assert_size_stride(primals_146, (32, ), (1, ))
    assert_size_stride(primals_147, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_153, (32, ), (1, ))
    assert_size_stride(primals_154, (32, ), (1, ))
    assert_size_stride(primals_155, (32, ), (1, ))
    assert_size_stride(primals_156, (32, ), (1, ))
    assert_size_stride(primals_157, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_158, (32, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (32, ), (1, ))
    assert_size_stride(primals_162, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_164, (32, ), (1, ))
    assert_size_stride(primals_165, (32, ), (1, ))
    assert_size_stride(primals_166, (32, ), (1, ))
    assert_size_stride(primals_167, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_168, (32, ), (1, ))
    assert_size_stride(primals_169, (32, ), (1, ))
    assert_size_stride(primals_170, (32, ), (1, ))
    assert_size_stride(primals_171, (32, ), (1, ))
    assert_size_stride(primals_172, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, ), (1, ))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 12, 2, 2), (48, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_1, buf0, 192, grid=grid(192), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 2, 2), (16, 4, 2, 1))
        buf2 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [batch_norm, sigmoid, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_1.run(buf3, buf1, primals_3, primals_4, primals_5, primals_6, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 8, 1, 1), (8, 1, 1, 1))
        buf5 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf6 = reinterpret_tensor(buf5, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_1, sigmoid_1, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2.run(buf6, buf4, primals_8, primals_9, primals_10, primals_11, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 4, 1, 1), (4, 1, 1, 1))
        buf8 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf9 = reinterpret_tensor(buf8, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_2, sigmoid_2, mul_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_3.run(buf9, buf7, primals_13, primals_14, primals_15, primals_16, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 1, 1), (4, 1, 1, 1))
        buf11 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf12 = reinterpret_tensor(buf11, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_3, sigmoid_3, mul_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_3.run(buf12, buf10, primals_18, primals_19, primals_20, primals_21, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 4, 1, 1), (4, 1, 1, 1))
        buf14 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf13, primals_23, primals_24, primals_25, primals_26, buf14, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf6, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 1, 1), (4, 1, 1, 1))
        buf16 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf15, primals_28, primals_29, primals_30, primals_31, buf16, 16, grid=grid(16), stream=stream0)
        buf17 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf9, buf14, buf16, buf17, 32, grid=grid(32), stream=stream0)
        del buf14
        del buf16
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 8, 1, 1), (8, 1, 1, 1))
        buf19 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf20 = reinterpret_tensor(buf19, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_6, sigmoid_6, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2.run(buf20, buf18, primals_33, primals_34, primals_35, primals_36, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_37, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 16, 1, 1), (16, 1, 1, 1))
        buf22 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf23 = reinterpret_tensor(buf22, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_7, sigmoid_7, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf23, buf21, primals_38, primals_39, primals_40, primals_41, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 8, 1, 1), (8, 1, 1, 1))
        buf25 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf26 = reinterpret_tensor(buf25, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_8, sigmoid_8, mul_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2.run(buf26, buf24, primals_43, primals_44, primals_45, primals_46, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 8, 1, 1), (8, 1, 1, 1))
        buf28 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf29 = reinterpret_tensor(buf28, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_9, sigmoid_9, mul_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2.run(buf29, buf27, primals_48, primals_49, primals_50, primals_51, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 8, 1, 1), (8, 1, 1, 1))
        buf31 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf32 = reinterpret_tensor(buf31, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_10, sigmoid_10, mul_10, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_7.run(buf32, buf30, primals_53, primals_54, primals_55, primals_56, buf26, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 8, 1, 1), (8, 1, 1, 1))
        buf34 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf35 = reinterpret_tensor(buf34, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_11, sigmoid_11, mul_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2.run(buf35, buf33, primals_58, primals_59, primals_60, primals_61, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 8, 1, 1), (8, 1, 1, 1))
        buf37 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf38 = reinterpret_tensor(buf37, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_12, sigmoid_12, mul_12, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_7.run(buf38, buf36, primals_63, primals_64, primals_65, primals_66, buf32, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 8, 1, 1), (8, 1, 1, 1))
        buf40 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf41 = reinterpret_tensor(buf40, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_13, sigmoid_13, mul_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_2.run(buf41, buf39, primals_68, primals_69, primals_70, primals_71, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 8, 1, 1), (8, 1, 1, 1))
        buf43 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_14], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf42, primals_73, primals_74, primals_75, primals_76, buf43, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf23, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 8, 1, 1), (8, 1, 1, 1))
        buf45 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_15], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf44, primals_78, primals_79, primals_80, primals_81, buf45, 32, grid=grid(32), stream=stream0)
        buf46 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf38, buf43, buf45, buf46, 64, grid=grid(64), stream=stream0)
        del buf43
        del buf45
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 16, 1, 1), (16, 1, 1, 1))
        buf48 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf49 = reinterpret_tensor(buf48, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_16, sigmoid_16, input_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf49, buf47, primals_83, primals_84, primals_85, primals_86, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_87, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 32, 1, 1), (32, 1, 1, 1))
        buf51 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf52 = reinterpret_tensor(buf51, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_17, sigmoid_17, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf52, buf50, primals_88, primals_89, primals_90, primals_91, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 16, 1, 1), (16, 1, 1, 1))
        buf54 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf55 = reinterpret_tensor(buf54, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_18, sigmoid_18, mul_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf55, buf53, primals_93, primals_94, primals_95, primals_96, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 16, 1, 1), (16, 1, 1, 1))
        buf57 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf58 = reinterpret_tensor(buf57, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_19, sigmoid_19, mul_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf58, buf56, primals_98, primals_99, primals_100, primals_101, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 16, 1, 1), (16, 1, 1, 1))
        buf60 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf61 = reinterpret_tensor(buf60, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_20, sigmoid_20, mul_20, input_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_11.run(buf61, buf59, primals_103, primals_104, primals_105, primals_106, buf55, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 16, 1, 1), (16, 1, 1, 1))
        buf63 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf64 = reinterpret_tensor(buf63, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_21, sigmoid_21, mul_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf64, buf62, primals_108, primals_109, primals_110, primals_111, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 16, 1, 1), (16, 1, 1, 1))
        buf66 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf67 = reinterpret_tensor(buf66, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_22, sigmoid_22, mul_22, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_sigmoid_11.run(buf67, buf65, primals_113, primals_114, primals_115, primals_116, buf61, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 16, 1, 1), (16, 1, 1, 1))
        buf69 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf70 = reinterpret_tensor(buf69, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_23, sigmoid_23, mul_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf70, buf68, primals_118, primals_119, primals_120, primals_121, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 16, 1, 1), (16, 1, 1, 1))
        buf72 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_24], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf71, primals_123, primals_124, primals_125, primals_126, buf72, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf52, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 16, 1, 1), (16, 1, 1, 1))
        buf74 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_25], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf73, primals_128, primals_129, primals_130, primals_131, buf74, 64, grid=grid(64), stream=stream0)
        buf75 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf67, buf72, buf74, buf75, 128, grid=grid(128), stream=stream0)
        del buf72
        del buf74
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 32, 1, 1), (32, 1, 1, 1))
        buf77 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf78 = reinterpret_tensor(buf77, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_26, sigmoid_26, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf78, buf76, primals_133, primals_134, primals_135, primals_136, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_137, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 64, 1, 1), (64, 1, 1, 1))
        buf80 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf81 = reinterpret_tensor(buf80, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_27, sigmoid_27, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf81, buf79, primals_138, primals_139, primals_140, primals_141, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 32, 1, 1), (32, 1, 1, 1))
        buf83 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf84 = reinterpret_tensor(buf83, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf83  # reuse
        buf85 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.int8)
        # Topologically Sorted Source Nodes: [batch_norm_28, sigmoid_28, x_1, max_pool2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_mul_sigmoid_15.run(buf84, buf82, primals_143, primals_144, primals_145, primals_146, buf85, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [max_pool2d_1], Original ATen: [aten.max_pool2d_with_indices]
        buf86 = torch.ops.aten.max_pool2d_with_indices.default(buf84, [9, 9], [1, 1], [4, 4])
        buf87 = buf86[0]
        buf88 = buf86[1]
        del buf86
        # Topologically Sorted Source Nodes: [max_pool2d_2], Original ATen: [aten.max_pool2d_with_indices]
        buf89 = torch.ops.aten.max_pool2d_with_indices.default(buf84, [13, 13], [1, 1], [6, 6])
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf92 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf84, buf87, buf90, buf92, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 64, 1, 1), (64, 1, 1, 1))
        buf94 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf95 = reinterpret_tensor(buf94, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_29, sigmoid_29, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf95, buf93, primals_148, primals_149, primals_150, primals_151, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 32, 1, 1), (32, 1, 1, 1))
        buf97 = reinterpret_tensor(buf90, (4, 32, 1, 1), (32, 1, 128, 128), 0); del buf90  # reuse
        buf98 = reinterpret_tensor(buf97, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_30, sigmoid_30, mul_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf98, buf96, primals_153, primals_154, primals_155, primals_156, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 32, 1, 1), (32, 1, 1, 1))
        buf100 = reinterpret_tensor(buf87, (4, 32, 1, 1), (32, 1, 128, 128), 0); del buf87  # reuse
        buf101 = reinterpret_tensor(buf100, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_31, sigmoid_31, mul_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf101, buf99, primals_158, primals_159, primals_160, primals_161, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 32, 1, 1), (32, 1, 1, 1))
        buf103 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_32], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf102, primals_163, primals_164, primals_165, primals_166, buf103, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf95, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 32, 1, 1), (32, 1, 1, 1))
        buf105 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_33], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf104, primals_168, primals_169, primals_170, primals_171, buf105, 128, grid=grid(128), stream=stream0)
        buf106 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf103, buf105, buf106, 256, grid=grid(256), stream=stream0)
        del buf103
        del buf105
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 64, 1, 1), (64, 1, 1, 1))
        buf108 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf109 = reinterpret_tensor(buf108, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_34, sigmoid_34, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf109, buf107, primals_173, primals_174, primals_175, primals_176, 256, grid=grid(256), stream=stream0)
    return (buf49, buf78, buf109, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, buf0, buf1, buf3, buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf15, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf29, buf30, buf32, buf33, buf35, buf36, buf38, buf39, buf41, buf42, buf44, buf46, buf47, buf49, buf50, buf52, buf53, buf55, buf56, buf58, buf59, buf61, buf62, buf64, buf65, buf67, buf68, buf70, buf71, buf73, buf75, buf76, buf78, buf79, buf81, buf82, buf84, buf85, buf88, buf91, buf92, buf93, buf95, buf96, buf98, buf99, buf101, buf102, buf104, buf106, buf107, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
