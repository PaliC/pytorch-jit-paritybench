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


# kernel path: inductor_cache/x6/cx6bvxvtvm36quxpvppzr64sxettuh7on4p2d4o6hbihulyvks5p.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_1 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_0 = async_compile.triton('triton_poi_fused_reflection_pad2d_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = ((xindex // 6) % 6)
    x2 = xindex // 36
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kq/ckql4crztf2qzchy6nm7nkp7vpkomimis5nxfle2ohcco6d7piqe.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_2 => convolution
#   input_3 => gt, mul, where
#   input_4 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_5]), kwargs = {})
triton_poi_fused_convolution_leaky_relu_reflection_pad2d_1 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_reflection_pad2d_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_reflection_pad2d_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_reflection_pad2d_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 5)
    x4 = xindex // 25
    x2 = ((xindex // 25) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + ((-1)*tl_math.abs((-2) + tl_math.abs((-1) + x0))) + ((-3)*tl_math.abs((-2) + tl_math.abs((-1) + x1))) + 9*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x5), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5h/c5hmvxmtsuodu62io5ru5z43jwptpmw2fighs2mblft34nkbz6nq.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_6 => add_1, mul_2, mul_3, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/gl/cglbcduienhwkesvwwwd7vv54es5id6ktxiduzrgdoh5bxex4sc4.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.leaky_relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_7 => gt_1, mul_4, where_1
#   input_8 => _unsafe_index_4, _unsafe_index_5
# Graph fragment:
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_1, %mul_4), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_1, [None, None, %sub_10, None]), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_4, [None, None, None, %sub_10]), kwargs = {})
triton_poi_fused_leaky_relu_reflection_pad2d_3 = async_compile.triton('triton_poi_fused_leaky_relu_reflection_pad2d_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_reflection_pad2d_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_reflection_pad2d_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + ((-1)*tl_math.abs((-1) + tl_math.abs((-1) + x0))) + ((-2)*tl_math.abs((-1) + tl_math.abs((-1) + x1))) + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/45/c45vuvr7nfs4ngmky4ycxcdnelpkk56vdsoa2qddgpczdlcspomp.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_10 => add_3, mul_6, mul_7, sub_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_11), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
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


# kernel path: inductor_cache/6b/c6b5av3urqxk25p6ahlwp7lugxq2gidsfvy6f5t7r4jeiu4epdyr.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.leaky_relu, aten._native_batch_norm_legit_no_training, aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_10 => add_3, mul_6, mul_7, sub_13
#   input_11 => add_4
#   input_12 => _unsafe_index_6, _unsafe_index_7
#   input_2 => convolution
#   input_3 => gt, mul, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_11), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %where), kwargs = {})
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_4, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_7 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_6, [None, None, None, %sub_5]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex // 25
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 5)
    x2 = ((xindex // 25) % 4)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (8 + ((-1)*tl_math.abs((-2) + tl_math.abs((-1) + x0))) + ((-3)*tl_math.abs((-2) + tl_math.abs((-1) + x1))) + 9*x5), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 0.2
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tmp9 = tmp0 + tmp8
    tl.store(out_ptr0 + (x6), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zh/czh2glc6w4hu2ul4iawszpijxp4ihccidajvqr2ifs2tbwytix4i.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_10, input_11, input_18, input_19, input_20], Original ATen: [aten.convolution, aten.leaky_relu, aten._native_batch_norm_legit_no_training, aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_10 => add_3, mul_6, mul_7, sub_13
#   input_11 => add_4
#   input_18 => add_8, mul_13, mul_14, sub_23
#   input_19 => add_9
#   input_2 => convolution
#   input_20 => _unsafe_index_10, _unsafe_index_11
#   input_3 => gt, mul, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_11), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %where), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %add_4), kwargs = {})
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_9, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_11 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_10, [None, None, None, %sub_5]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex // 25
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 5)
    x2 = ((xindex // 25) % 4)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (8 + ((-1)*tl_math.abs((-2) + tl_math.abs((-1) + x0))) + ((-3)*tl_math.abs((-2) + tl_math.abs((-1) + x1))) + 9*x5), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.2
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tmp1 + tmp9
    tmp11 = tmp0 + tmp10
    tl.store(out_ptr0 + (x6), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uq/cuqhzuczqg3z6ewdkw6caotviuz5fpcq4ww3n7btll57275a6ygi.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_10, input_11, input_18, input_19, input_26, input_27], Original ATen: [aten.convolution, aten.leaky_relu, aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   input_10 => add_3, mul_6, mul_7, sub_13
#   input_11 => add_4
#   input_18 => add_8, mul_13, mul_14, sub_23
#   input_19 => add_9
#   input_2 => convolution
#   input_26 => add_13, mul_20, mul_21, sub_33
#   input_27 => add_14
#   input_3 => gt, mul, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_11), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %where), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %add_4), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_41), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_43), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_45), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_47), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %add_9), kwargs = {})
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_leaky_relu_backward_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_leaky_relu_backward_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_leaky_relu_backward_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_leaky_relu_backward_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 9
    x4 = xindex
    x1 = ((xindex // 9) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x4), xmask)
    tmp4 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 0.2
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tmp2 + tmp10
    tmp12 = tmp1 + tmp11
    tmp13 = tmp0 + tmp12
    tmp14 = tmp10 > tmp6
    tl.store(out_ptr0 + (x4), tmp13, xmask)
    tl.store(out_ptr1 + (x4), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wm/cwm4h5ne7plspo5iuolrcqcfm4i3yrhuersln5awex2pbti77c2u.py
# Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_28 => _unsafe_index_14, _unsafe_index_15
# Graph fragment:
#   %_unsafe_index_14 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_14, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_15 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_14, [None, None, None, %sub_5]), kwargs = {})
triton_poi_fused_reflection_pad2d_8 = async_compile.triton('triton_poi_fused_reflection_pad2d_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 5)
    x2 = xindex // 25
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + ((-1)*tl_math.abs((-2) + tl_math.abs((-1) + x0))) + ((-3)*tl_math.abs((-2) + tl_math.abs((-1) + x1))) + 9*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pc/cpcpfmu2mybi3jbmd4unmz5wnrppsjft36sreyhynrvgojenrmph.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_34 => add_18, mul_27, mul_28, sub_43
#   input_35 => add_19
#   input_36 => _unsafe_index_18, _unsafe_index_19
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_57), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_59), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_61), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %unsqueeze_63), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %add_14), kwargs = {})
#   %_unsafe_index_18 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_19, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_19 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_18, [None, None, None, %sub_5]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_reflection_pad2d_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_reflection_pad2d_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_reflection_pad2d_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_reflection_pad2d_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 25
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 5)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (8 + ((-1)*tl_math.abs((-2) + tl_math.abs((-1) + x0))) + ((-3)*tl_math.abs((-2) + tl_math.abs((-1) + x1))) + 9*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kn/ckn7gwly4z65igv2azgfrz22ayrjfv2rxybi6lvw3lkxpkphcgmo.py
# Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_37 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_19, %primals_44, %primals_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iy/ciyiiunjidv2nc3qslsjo276jbyllfkadm5u7ijywbgvrold3fj6.py
# Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_38 => add_21, mul_30, mul_31, sub_48
#   input_39 => _unsafe_index_20, _unsafe_index_21
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_65), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_67), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %unsqueeze_69), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_71), kwargs = {})
#   %_unsafe_index_20 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_21, [None, None, %sub_10, None]), kwargs = {})
#   %_unsafe_index_21 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_20, [None, None, None, %sub_10]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x4 = xindex // 16
    x2 = ((xindex // 16) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + ((-1)*tl_math.abs((-1) + tl_math.abs((-1) + x0))) + ((-2)*tl_math.abs((-1) + tl_math.abs((-1) + x1))) + 4*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bl/cblspnfsfiltvdunkugkr5xpvilpuvsij4el7ihsl72wiluvqeku.py
# Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten.convolution, aten.sigmoid]
# Source node to ATen node mapping:
#   input_40 => convolution_10
#   input_41 => sigmoid
# Graph fragment:
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_21, %primals_50, %primals_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_10,), kwargs = {})
triton_poi_fused_convolution_sigmoid_12 = async_compile.triton('triton_poi_fused_convolution_sigmoid_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_sigmoid_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_sigmoid_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, ), (1, ))
    assert_size_stride(primals_38, (4, ), (1, ))
    assert_size_stride(primals_39, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_40, (4, ), (1, ))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, ), (1, ))
    assert_size_stride(primals_43, (4, ), (1, ))
    assert_size_stride(primals_44, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (4, ), (1, ))
    assert_size_stride(primals_50, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_51, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0.run(primals_1, buf0, 576, grid=grid(576), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 3, 3), (36, 9, 3, 1))
        buf2 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_reflection_pad2d_1.run(buf1, primals_3, buf2, 400, grid=grid(400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 2, 2), (16, 4, 2, 1))
        buf4 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf3, primals_5, primals_6, primals_7, primals_8, buf4, 64, grid=grid(64), stream=stream0)
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.leaky_relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_reflection_pad2d_3.run(buf4, buf5, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 1, 1), (4, 1, 1, 1))
        buf7 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf6, primals_10, primals_11, primals_12, primals_13, buf7, 16, grid=grid(16), stream=stream0)
        del primals_13
        buf8 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3, input_10, input_11, input_12], Original ATen: [aten.convolution, aten.leaky_relu, aten._native_batch_norm_legit_no_training, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_5.run(buf7, buf1, primals_3, buf8, 400, grid=grid(400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 4, 2, 2), (16, 4, 2, 1))
        buf10 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf9, primals_15, primals_16, primals_17, primals_18, buf10, 64, grid=grid(64), stream=stream0)
        buf11 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.leaky_relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_reflection_pad2d_3.run(buf10, buf11, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 4, 1, 1), (4, 1, 1, 1))
        buf13 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf12, primals_20, primals_21, primals_22, primals_23, buf13, 16, grid=grid(16), stream=stream0)
        del primals_23
        buf14 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3, input_10, input_11, input_18, input_19, input_20], Original ATen: [aten.convolution, aten.leaky_relu, aten._native_batch_norm_legit_no_training, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_reflection_pad2d_6.run(buf13, buf7, buf1, primals_3, buf14, 400, grid=grid(400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 2, 2), (16, 4, 2, 1))
        buf16 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf15, primals_25, primals_26, primals_27, primals_28, buf16, 64, grid=grid(64), stream=stream0)
        buf17 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten.leaky_relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_reflection_pad2d_3.run(buf16, buf17, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 4, 1, 1), (4, 1, 1, 1))
        buf19 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf18, primals_30, primals_31, primals_32, primals_33, buf19, 16, grid=grid(16), stream=stream0)
        del primals_33
        buf20 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        buf33 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_2, input_3, input_10, input_11, input_18, input_19, input_26, input_27], Original ATen: [aten.convolution, aten.leaky_relu, aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_leaky_relu_leaky_relu_backward_7.run(buf19, buf13, buf7, buf1, primals_3, buf20, buf33, 144, grid=grid(144), stream=stream0)
        del buf1
        del buf13
        del buf19
        del primals_3
        buf21 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_8.run(buf20, buf21, 400, grid=grid(400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 4, 2, 2), (16, 4, 2, 1))
        buf23 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf22, primals_35, primals_36, primals_37, primals_38, buf23, 64, grid=grid(64), stream=stream0)
        buf24 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.leaky_relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_reflection_pad2d_3.run(buf23, buf24, 256, grid=grid(256), stream=stream0)
        del buf23
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_39, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 4, 1, 1), (4, 1, 1, 1))
        buf26 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf25, primals_40, primals_41, primals_42, primals_43, buf26, 16, grid=grid(16), stream=stream0)
        del primals_43
        buf27 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_reflection_pad2d_9.run(buf26, buf20, buf27, 400, grid=grid(400), stream=stream0)
        del buf20
        del buf26
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 4, 2, 2), (16, 4, 2, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(buf29, primals_45, 64, grid=grid(64), stream=stream0)
        del primals_45
        buf30 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_reflection_pad2d_11.run(buf29, primals_46, primals_47, primals_48, primals_49, buf30, 256, grid=grid(256), stream=stream0)
        del primals_49
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 4, 1, 1), (4, 1, 1, 1))
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten.convolution, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_sigmoid_12.run(buf32, primals_51, 16, grid=grid(16), stream=stream0)
        del primals_51
    return (buf32, primals_2, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_50, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf15, buf17, buf18, buf21, buf22, buf24, buf25, buf27, buf29, buf30, buf32, buf33, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
