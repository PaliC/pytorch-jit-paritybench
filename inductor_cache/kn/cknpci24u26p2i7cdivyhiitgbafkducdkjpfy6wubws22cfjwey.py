# AOT ID: ['26_forward']
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


# kernel path: inductor_cache/3a/c3azsetduzbi725lopx3345l66sjenoycxbvorhy4dwr5tjwlgpf.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   interpolate => add_93, add_94, convert_element_type, convert_element_type_1, iota, mul_368, mul_369
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_368, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_93, torch.float32), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_369 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_369, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_0 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3y/c3ybqplqwqos3pc3be53kailmn3meskeno2ksmc4cz5kk3cxputn.py
# Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   interpolate_1 => add_97, add_98, convert_element_type_4, convert_element_type_5, iota_2, mul_373, mul_374
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_373, 0), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_97, torch.float32), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_98, 0.5), kwargs = {})
#   %convert_element_type_5 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_374, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_1 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qr/cqrub5rxduws2ezszumvxrloxhjp7gp2tdx6vxtyzjm4jdywcxuo.py
# Topologically Sorted Source Nodes: [fea], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   fea => convolution
# Graph fragment:
#   %convolution : [num_users=9] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/mf/cmfsgutoskb452o2djyqrz6jm34dgjnfzjgnhclgpyo65wn7i7pm.py
# Topologically Sorted Source Nodes: [conv2d_1, x1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   x1 => gt, mul, where
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.2), kwargs = {})
#   %where : [num_users=5] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %mul), kwargs = {})
#   %gt_557 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7 > tmp3
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/em/cemtp2accrm5itj5jh7snx27pa3iacxvicgvqtoatj5vtt3episb.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 96)
    x0 = (xindex % 4096)
    x2 = xindex // 393216
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-64) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuulo25uhvrkxmae557cg27hnkwrk4ndsfw2loy56ikriucupc3x.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where, %where_1], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 128)
    x0 = (xindex % 4096)
    x2 = xindex // 524288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-64) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr3 + (x0 + 4096*((-96) + x1) + 131072*x2), tmp20, other=0.0)
    tmp24 = tl.load(in_ptr4 + ((-96) + x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp28 = 0.2
    tmp29 = tmp25 * tmp28
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp20, tmp30, tmp31)
    tmp33 = tl.where(tmp9, tmp19, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tl.store(out_ptr0 + (x3), tmp34, None)
''', device_str='cuda')


# kernel path: inductor_cache/ie/ciekqnfqhz4ru5rttpe6ptxygkn4nyb6qnvtvxywmjo6mz7a7bgj.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where, %where_1, %where_2], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 160)
    x0 = (xindex % 4096)
    x2 = xindex // 655360
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-64) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr3 + (x0 + 4096*((-96) + x1) + 131072*x2), tmp23, other=0.0)
    tmp25 = tl.load(in_ptr4 + ((-96) + x1), tmp23, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp26 > tmp27
    tmp29 = 0.2
    tmp30 = tmp26 * tmp29
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp0 >= tmp21
    tmp35 = tl.full([1], 160, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr5 + (x0 + 4096*((-128) + x1) + 131072*x2), tmp34, other=0.0)
    tmp38 = tl.load(in_ptr6 + ((-128) + x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = 0.0
    tmp41 = tmp39 > tmp40
    tmp42 = 0.2
    tmp43 = tmp39 * tmp42
    tmp44 = tl.where(tmp41, tmp39, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp34, tmp44, tmp45)
    tmp47 = tl.where(tmp23, tmp33, tmp46)
    tmp48 = tl.where(tmp9, tmp19, tmp47)
    tmp49 = tl.where(tmp4, tmp5, tmp48)
    tl.store(out_ptr0 + (x3), tmp49, None)
''', device_str='cuda')


# kernel path: inductor_cache/x5/cx54axs55osl2u5lsv4gj2rksfhifgmydmegoj6senhpemj55aoq.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where, %where_1, %where_2, %where_3], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 192)
    x0 = (xindex % 4096)
    x2 = xindex // 786432
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-64) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr3 + (x0 + 4096*((-96) + x1) + 131072*x2), tmp23, other=0.0)
    tmp25 = tl.load(in_ptr4 + ((-96) + x1), tmp23, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp26 > tmp27
    tmp29 = 0.2
    tmp30 = tmp26 * tmp29
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp0 >= tmp21
    tmp35 = tl.full([1], 160, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr5 + (x0 + 4096*((-128) + x1) + 131072*x2), tmp37, other=0.0)
    tmp39 = tl.load(in_ptr6 + ((-128) + x1), tmp37, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 + tmp39
    tmp41 = 0.0
    tmp42 = tmp40 > tmp41
    tmp43 = 0.2
    tmp44 = tmp40 * tmp43
    tmp45 = tl.where(tmp42, tmp40, tmp44)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp37, tmp45, tmp46)
    tmp48 = tmp0 >= tmp35
    tmp49 = tl.full([1], 192, tl.int64)
    tmp50 = tmp0 < tmp49
    tmp51 = tl.load(in_ptr7 + (x0 + 4096*((-160) + x1) + 131072*x2), tmp48, other=0.0)
    tmp52 = tl.load(in_ptr8 + ((-160) + x1), tmp48, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = 0.0
    tmp55 = tmp53 > tmp54
    tmp56 = 0.2
    tmp57 = tmp53 * tmp56
    tmp58 = tl.where(tmp55, tmp53, tmp57)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp48, tmp58, tmp59)
    tmp61 = tl.where(tmp37, tmp47, tmp60)
    tmp62 = tl.where(tmp23, tmp33, tmp61)
    tmp63 = tl.where(tmp9, tmp19, tmp62)
    tmp64 = tl.where(tmp4, tmp5, tmp63)
    tl.store(out_ptr0 + (x3), tmp64, None)
''', device_str='cuda')


# kernel path: inductor_cache/co/ccol2b3kza2seoq3jxmukubcbwftqep4mwbog42ya4ujumdlxknj.py
# Topologically Sorted Source Nodes: [x5, mul, out], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul => mul_4
#   out => add
#   x5 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_12, %primals_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_5, 0.2), kwargs = {})
#   %add : [num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_8 = async_compile.triton('triton_poi_fused_add_convolution_mul_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/2c/c2czd662lsi675xiyubk5q72h6xz2e77eoury4l2yo55xkrnkpm2.py
# Topologically Sorted Source Nodes: [x5_2, mul_2, out_2, mul_3, input_1], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_1 => add_3
#   mul_2 => mul_14
#   mul_3 => mul_15
#   out_2 => add_2
#   x5_2 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 0.2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %add_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.2), kwargs = {})
#   %add_3 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_9 = async_compile.triton('triton_poi_fused_add_convolution_mul_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp3
    tmp9 = tmp7 + tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/ko/ckomm2ybs7gy65qbwlh6wphh6uwfnwb6rwjupcwtys3nxafpo445.py
# Topologically Sorted Source Nodes: [trunk, fea_1, interpolate], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
# Source node to ATen node mapping:
#   fea_1 => add_92
#   interpolate => _unsafe_index
#   trunk => convolution_346
# Graph fragment:
#   %convolution_346 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_91, %primals_694, %primals_695, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_346), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_92, [None, None, %unsqueeze, %convert_element_type_1]), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_10 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 128) % 128)
    x0 = (xindex % 128)
    x5 = xindex // 16384
    x2 = ((xindex // 16384) % 64)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 64*tmp4 + 4096*x5), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 64*tmp4 + 4096*x5), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tl.store(out_ptr0 + (x6), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/my/cmyfstqeymxmpb7xo2swef4s5qg3z5lpgkd44iqhwkieoxt5jyhu.py
# Topologically Sorted Source Nodes: [conv2d_347, fea_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   conv2d_347 => convolution_347
#   fea_2 => gt_276, mul_372, where_276
# Graph fragment:
#   %convolution_347 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index, %primals_696, %primals_697, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_276 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_347, 0), kwargs = {})
#   %mul_372 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_347, 0.2), kwargs = {})
#   %where_276 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_276, %convolution_347, %mul_372), kwargs = {})
#   %gt_281 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_276, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_11 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16384) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7 > tmp3
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwrmq3khqw65ykl5ok7las7xtgpcoflif77c3lw2fubyhuckp6dq.py
# Topologically Sorted Source Nodes: [conv2d_347, fea_2, interpolate_1], Original ATen: [aten.convolution, aten.leaky_relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   conv2d_347 => convolution_347
#   fea_2 => gt_276, mul_372, where_276
#   interpolate_1 => _unsafe_index_1
# Graph fragment:
#   %convolution_347 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index, %primals_696, %primals_697, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_276 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_347, 0), kwargs = {})
#   %mul_372 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_347, 0.2), kwargs = {})
#   %where_276 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_276, %convolution_347, %mul_372), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%where_276, [None, None, %unsqueeze_1, %convert_element_type_5]), kwargs = {})
triton_poi_fused__unsafe_index_convolution_leaky_relu_12 = async_compile.triton('triton_poi_fused__unsafe_index_convolution_leaky_relu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_convolution_leaky_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_convolution_leaky_relu_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    x5 = xindex // 65536
    x2 = ((xindex // 65536) % 64)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 128*tmp4 + 16384*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tl.store(out_ptr0 + (x6), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/y4/cy4erinr43ep6knexjoo3sffflxtvmzbjugiykj65f5v6e5w3jv6.py
# Topologically Sorted Source Nodes: [conv2d_348, fea_3], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_348 => convolution_348
#   fea_3 => gt_277, mul_377, where_277
# Graph fragment:
#   %convolution_348 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_698, %primals_699, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_277 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_348, 0), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_348, 0.2), kwargs = {})
#   %where_277 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_277, %convolution_348, %mul_377), kwargs = {})
triton_poi_fused_convolution_leaky_relu_13 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 65536) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/bd/cbdiaarnq6q2w5lebm4hfl4vqqj3w6s4ycyt7kydbcaum2fqz2cc.py
# Topologically Sorted Source Nodes: [out_69], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_69 => convolution_350
# Graph fragment:
#   %convolution_350 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_278, %primals_702, %primals_703, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 65536) % 3)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_31, (32, ), (1, ))
    assert_size_stride(primals_32, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_37, (32, ), (1, ))
    assert_size_stride(primals_38, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_39, (32, ), (1, ))
    assert_size_stride(primals_40, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_41, (32, ), (1, ))
    assert_size_stride(primals_42, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_48, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_49, (32, ), (1, ))
    assert_size_stride(primals_50, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_55, (32, ), (1, ))
    assert_size_stride(primals_56, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_57, (32, ), (1, ))
    assert_size_stride(primals_58, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_67, (32, ), (1, ))
    assert_size_stride(primals_68, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_69, (32, ), (1, ))
    assert_size_stride(primals_70, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_71, (32, ), (1, ))
    assert_size_stride(primals_72, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_75, (32, ), (1, ))
    assert_size_stride(primals_76, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_77, (32, ), (1, ))
    assert_size_stride(primals_78, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_79, (32, ), (1, ))
    assert_size_stride(primals_80, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_81, (32, ), (1, ))
    assert_size_stride(primals_82, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_86, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_87, (32, ), (1, ))
    assert_size_stride(primals_88, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_89, (32, ), (1, ))
    assert_size_stride(primals_90, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_91, (32, ), (1, ))
    assert_size_stride(primals_92, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_97, (32, ), (1, ))
    assert_size_stride(primals_98, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_115, (32, ), (1, ))
    assert_size_stride(primals_116, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_125, (32, ), (1, ))
    assert_size_stride(primals_126, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_127, (32, ), (1, ))
    assert_size_stride(primals_128, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_129, (32, ), (1, ))
    assert_size_stride(primals_130, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_131, (32, ), (1, ))
    assert_size_stride(primals_132, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_137, (32, ), (1, ))
    assert_size_stride(primals_138, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_139, (32, ), (1, ))
    assert_size_stride(primals_140, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_141, (32, ), (1, ))
    assert_size_stride(primals_142, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_145, (32, ), (1, ))
    assert_size_stride(primals_146, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_147, (32, ), (1, ))
    assert_size_stride(primals_148, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_149, (32, ), (1, ))
    assert_size_stride(primals_150, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_151, (32, ), (1, ))
    assert_size_stride(primals_152, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_155, (32, ), (1, ))
    assert_size_stride(primals_156, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_157, (32, ), (1, ))
    assert_size_stride(primals_158, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_161, (32, ), (1, ))
    assert_size_stride(primals_162, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_165, (32, ), (1, ))
    assert_size_stride(primals_166, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_167, (32, ), (1, ))
    assert_size_stride(primals_168, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_169, (32, ), (1, ))
    assert_size_stride(primals_170, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_171, (32, ), (1, ))
    assert_size_stride(primals_172, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_175, (32, ), (1, ))
    assert_size_stride(primals_176, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_177, (32, ), (1, ))
    assert_size_stride(primals_178, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_179, (32, ), (1, ))
    assert_size_stride(primals_180, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_181, (32, ), (1, ))
    assert_size_stride(primals_182, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_183, (64, ), (1, ))
    assert_size_stride(primals_184, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_185, (32, ), (1, ))
    assert_size_stride(primals_186, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_187, (32, ), (1, ))
    assert_size_stride(primals_188, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_191, (32, ), (1, ))
    assert_size_stride(primals_192, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_193, (64, ), (1, ))
    assert_size_stride(primals_194, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_195, (32, ), (1, ))
    assert_size_stride(primals_196, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_197, (32, ), (1, ))
    assert_size_stride(primals_198, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_205, (32, ), (1, ))
    assert_size_stride(primals_206, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_207, (32, ), (1, ))
    assert_size_stride(primals_208, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_213, (64, ), (1, ))
    assert_size_stride(primals_214, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_217, (32, ), (1, ))
    assert_size_stride(primals_218, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_219, (32, ), (1, ))
    assert_size_stride(primals_220, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_221, (32, ), (1, ))
    assert_size_stride(primals_222, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_225, (32, ), (1, ))
    assert_size_stride(primals_226, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_227, (32, ), (1, ))
    assert_size_stride(primals_228, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_229, (32, ), (1, ))
    assert_size_stride(primals_230, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_231, (32, ), (1, ))
    assert_size_stride(primals_232, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_233, (64, ), (1, ))
    assert_size_stride(primals_234, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_235, (32, ), (1, ))
    assert_size_stride(primals_236, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_237, (32, ), (1, ))
    assert_size_stride(primals_238, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_239, (32, ), (1, ))
    assert_size_stride(primals_240, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_241, (32, ), (1, ))
    assert_size_stride(primals_242, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_243, (64, ), (1, ))
    assert_size_stride(primals_244, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_245, (32, ), (1, ))
    assert_size_stride(primals_246, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_247, (32, ), (1, ))
    assert_size_stride(primals_248, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_249, (32, ), (1, ))
    assert_size_stride(primals_250, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_251, (32, ), (1, ))
    assert_size_stride(primals_252, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_255, (32, ), (1, ))
    assert_size_stride(primals_256, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_257, (32, ), (1, ))
    assert_size_stride(primals_258, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_259, (32, ), (1, ))
    assert_size_stride(primals_260, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_261, (32, ), (1, ))
    assert_size_stride(primals_262, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_263, (64, ), (1, ))
    assert_size_stride(primals_264, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_267, (32, ), (1, ))
    assert_size_stride(primals_268, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_269, (32, ), (1, ))
    assert_size_stride(primals_270, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_271, (32, ), (1, ))
    assert_size_stride(primals_272, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_275, (32, ), (1, ))
    assert_size_stride(primals_276, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_277, (32, ), (1, ))
    assert_size_stride(primals_278, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_279, (32, ), (1, ))
    assert_size_stride(primals_280, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_281, (32, ), (1, ))
    assert_size_stride(primals_282, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_283, (64, ), (1, ))
    assert_size_stride(primals_284, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_285, (32, ), (1, ))
    assert_size_stride(primals_286, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_287, (32, ), (1, ))
    assert_size_stride(primals_288, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_291, (32, ), (1, ))
    assert_size_stride(primals_292, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_293, (64, ), (1, ))
    assert_size_stride(primals_294, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_295, (32, ), (1, ))
    assert_size_stride(primals_296, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_297, (32, ), (1, ))
    assert_size_stride(primals_298, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_299, (32, ), (1, ))
    assert_size_stride(primals_300, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_301, (32, ), (1, ))
    assert_size_stride(primals_302, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_303, (64, ), (1, ))
    assert_size_stride(primals_304, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_305, (32, ), (1, ))
    assert_size_stride(primals_306, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_307, (32, ), (1, ))
    assert_size_stride(primals_308, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_309, (32, ), (1, ))
    assert_size_stride(primals_310, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_311, (32, ), (1, ))
    assert_size_stride(primals_312, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_315, (32, ), (1, ))
    assert_size_stride(primals_316, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_317, (32, ), (1, ))
    assert_size_stride(primals_318, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_319, (32, ), (1, ))
    assert_size_stride(primals_320, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_321, (32, ), (1, ))
    assert_size_stride(primals_322, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_323, (64, ), (1, ))
    assert_size_stride(primals_324, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_325, (32, ), (1, ))
    assert_size_stride(primals_326, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_327, (32, ), (1, ))
    assert_size_stride(primals_328, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_329, (32, ), (1, ))
    assert_size_stride(primals_330, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_331, (32, ), (1, ))
    assert_size_stride(primals_332, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_333, (64, ), (1, ))
    assert_size_stride(primals_334, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_335, (32, ), (1, ))
    assert_size_stride(primals_336, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_337, (32, ), (1, ))
    assert_size_stride(primals_338, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_339, (32, ), (1, ))
    assert_size_stride(primals_340, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_341, (32, ), (1, ))
    assert_size_stride(primals_342, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_345, (32, ), (1, ))
    assert_size_stride(primals_346, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_347, (32, ), (1, ))
    assert_size_stride(primals_348, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_349, (32, ), (1, ))
    assert_size_stride(primals_350, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_351, (32, ), (1, ))
    assert_size_stride(primals_352, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_353, (64, ), (1, ))
    assert_size_stride(primals_354, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_355, (32, ), (1, ))
    assert_size_stride(primals_356, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_357, (32, ), (1, ))
    assert_size_stride(primals_358, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_359, (32, ), (1, ))
    assert_size_stride(primals_360, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_361, (32, ), (1, ))
    assert_size_stride(primals_362, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_363, (64, ), (1, ))
    assert_size_stride(primals_364, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_365, (32, ), (1, ))
    assert_size_stride(primals_366, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_367, (32, ), (1, ))
    assert_size_stride(primals_368, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_369, (32, ), (1, ))
    assert_size_stride(primals_370, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_371, (32, ), (1, ))
    assert_size_stride(primals_372, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_373, (64, ), (1, ))
    assert_size_stride(primals_374, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_375, (32, ), (1, ))
    assert_size_stride(primals_376, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_377, (32, ), (1, ))
    assert_size_stride(primals_378, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_379, (32, ), (1, ))
    assert_size_stride(primals_380, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_381, (32, ), (1, ))
    assert_size_stride(primals_382, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_383, (64, ), (1, ))
    assert_size_stride(primals_384, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_385, (32, ), (1, ))
    assert_size_stride(primals_386, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_387, (32, ), (1, ))
    assert_size_stride(primals_388, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_389, (32, ), (1, ))
    assert_size_stride(primals_390, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_391, (32, ), (1, ))
    assert_size_stride(primals_392, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_393, (64, ), (1, ))
    assert_size_stride(primals_394, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_395, (32, ), (1, ))
    assert_size_stride(primals_396, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_397, (32, ), (1, ))
    assert_size_stride(primals_398, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_399, (32, ), (1, ))
    assert_size_stride(primals_400, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_401, (32, ), (1, ))
    assert_size_stride(primals_402, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_403, (64, ), (1, ))
    assert_size_stride(primals_404, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_405, (32, ), (1, ))
    assert_size_stride(primals_406, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_407, (32, ), (1, ))
    assert_size_stride(primals_408, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_409, (32, ), (1, ))
    assert_size_stride(primals_410, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_411, (32, ), (1, ))
    assert_size_stride(primals_412, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_413, (64, ), (1, ))
    assert_size_stride(primals_414, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_415, (32, ), (1, ))
    assert_size_stride(primals_416, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_417, (32, ), (1, ))
    assert_size_stride(primals_418, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_419, (32, ), (1, ))
    assert_size_stride(primals_420, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_421, (32, ), (1, ))
    assert_size_stride(primals_422, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_423, (64, ), (1, ))
    assert_size_stride(primals_424, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_425, (32, ), (1, ))
    assert_size_stride(primals_426, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_427, (32, ), (1, ))
    assert_size_stride(primals_428, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_429, (32, ), (1, ))
    assert_size_stride(primals_430, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_431, (32, ), (1, ))
    assert_size_stride(primals_432, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_433, (64, ), (1, ))
    assert_size_stride(primals_434, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_435, (32, ), (1, ))
    assert_size_stride(primals_436, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_437, (32, ), (1, ))
    assert_size_stride(primals_438, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_439, (32, ), (1, ))
    assert_size_stride(primals_440, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_441, (32, ), (1, ))
    assert_size_stride(primals_442, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_443, (64, ), (1, ))
    assert_size_stride(primals_444, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_445, (32, ), (1, ))
    assert_size_stride(primals_446, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_447, (32, ), (1, ))
    assert_size_stride(primals_448, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_449, (32, ), (1, ))
    assert_size_stride(primals_450, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_451, (32, ), (1, ))
    assert_size_stride(primals_452, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_453, (64, ), (1, ))
    assert_size_stride(primals_454, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_455, (32, ), (1, ))
    assert_size_stride(primals_456, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_457, (32, ), (1, ))
    assert_size_stride(primals_458, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_459, (32, ), (1, ))
    assert_size_stride(primals_460, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_461, (32, ), (1, ))
    assert_size_stride(primals_462, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_463, (64, ), (1, ))
    assert_size_stride(primals_464, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_465, (32, ), (1, ))
    assert_size_stride(primals_466, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_467, (32, ), (1, ))
    assert_size_stride(primals_468, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_469, (32, ), (1, ))
    assert_size_stride(primals_470, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_471, (32, ), (1, ))
    assert_size_stride(primals_472, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_473, (64, ), (1, ))
    assert_size_stride(primals_474, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_475, (32, ), (1, ))
    assert_size_stride(primals_476, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_477, (32, ), (1, ))
    assert_size_stride(primals_478, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_479, (32, ), (1, ))
    assert_size_stride(primals_480, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_481, (32, ), (1, ))
    assert_size_stride(primals_482, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_483, (64, ), (1, ))
    assert_size_stride(primals_484, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_485, (32, ), (1, ))
    assert_size_stride(primals_486, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_487, (32, ), (1, ))
    assert_size_stride(primals_488, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_489, (32, ), (1, ))
    assert_size_stride(primals_490, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_491, (32, ), (1, ))
    assert_size_stride(primals_492, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_493, (64, ), (1, ))
    assert_size_stride(primals_494, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_495, (32, ), (1, ))
    assert_size_stride(primals_496, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_497, (32, ), (1, ))
    assert_size_stride(primals_498, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_499, (32, ), (1, ))
    assert_size_stride(primals_500, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_501, (32, ), (1, ))
    assert_size_stride(primals_502, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_503, (64, ), (1, ))
    assert_size_stride(primals_504, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_505, (32, ), (1, ))
    assert_size_stride(primals_506, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_507, (32, ), (1, ))
    assert_size_stride(primals_508, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_509, (32, ), (1, ))
    assert_size_stride(primals_510, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_511, (32, ), (1, ))
    assert_size_stride(primals_512, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_513, (64, ), (1, ))
    assert_size_stride(primals_514, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_515, (32, ), (1, ))
    assert_size_stride(primals_516, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_517, (32, ), (1, ))
    assert_size_stride(primals_518, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_519, (32, ), (1, ))
    assert_size_stride(primals_520, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_521, (32, ), (1, ))
    assert_size_stride(primals_522, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_523, (64, ), (1, ))
    assert_size_stride(primals_524, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_525, (32, ), (1, ))
    assert_size_stride(primals_526, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_527, (32, ), (1, ))
    assert_size_stride(primals_528, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_529, (32, ), (1, ))
    assert_size_stride(primals_530, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_531, (32, ), (1, ))
    assert_size_stride(primals_532, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_533, (64, ), (1, ))
    assert_size_stride(primals_534, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_535, (32, ), (1, ))
    assert_size_stride(primals_536, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_537, (32, ), (1, ))
    assert_size_stride(primals_538, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_539, (32, ), (1, ))
    assert_size_stride(primals_540, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_541, (32, ), (1, ))
    assert_size_stride(primals_542, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_543, (64, ), (1, ))
    assert_size_stride(primals_544, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_545, (32, ), (1, ))
    assert_size_stride(primals_546, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_547, (32, ), (1, ))
    assert_size_stride(primals_548, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_549, (32, ), (1, ))
    assert_size_stride(primals_550, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_551, (32, ), (1, ))
    assert_size_stride(primals_552, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_553, (64, ), (1, ))
    assert_size_stride(primals_554, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_555, (32, ), (1, ))
    assert_size_stride(primals_556, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_557, (32, ), (1, ))
    assert_size_stride(primals_558, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_559, (32, ), (1, ))
    assert_size_stride(primals_560, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_561, (32, ), (1, ))
    assert_size_stride(primals_562, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_563, (64, ), (1, ))
    assert_size_stride(primals_564, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_565, (32, ), (1, ))
    assert_size_stride(primals_566, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_567, (32, ), (1, ))
    assert_size_stride(primals_568, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_569, (32, ), (1, ))
    assert_size_stride(primals_570, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_571, (32, ), (1, ))
    assert_size_stride(primals_572, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_573, (64, ), (1, ))
    assert_size_stride(primals_574, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_575, (32, ), (1, ))
    assert_size_stride(primals_576, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_577, (32, ), (1, ))
    assert_size_stride(primals_578, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_579, (32, ), (1, ))
    assert_size_stride(primals_580, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_581, (32, ), (1, ))
    assert_size_stride(primals_582, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_583, (64, ), (1, ))
    assert_size_stride(primals_584, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_585, (32, ), (1, ))
    assert_size_stride(primals_586, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_587, (32, ), (1, ))
    assert_size_stride(primals_588, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_589, (32, ), (1, ))
    assert_size_stride(primals_590, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_591, (32, ), (1, ))
    assert_size_stride(primals_592, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_593, (64, ), (1, ))
    assert_size_stride(primals_594, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_595, (32, ), (1, ))
    assert_size_stride(primals_596, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_597, (32, ), (1, ))
    assert_size_stride(primals_598, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_599, (32, ), (1, ))
    assert_size_stride(primals_600, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_601, (32, ), (1, ))
    assert_size_stride(primals_602, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_603, (64, ), (1, ))
    assert_size_stride(primals_604, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_605, (32, ), (1, ))
    assert_size_stride(primals_606, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_607, (32, ), (1, ))
    assert_size_stride(primals_608, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_609, (32, ), (1, ))
    assert_size_stride(primals_610, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_611, (32, ), (1, ))
    assert_size_stride(primals_612, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_613, (64, ), (1, ))
    assert_size_stride(primals_614, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_615, (32, ), (1, ))
    assert_size_stride(primals_616, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_617, (32, ), (1, ))
    assert_size_stride(primals_618, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_619, (32, ), (1, ))
    assert_size_stride(primals_620, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_621, (32, ), (1, ))
    assert_size_stride(primals_622, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_623, (64, ), (1, ))
    assert_size_stride(primals_624, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_625, (32, ), (1, ))
    assert_size_stride(primals_626, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_627, (32, ), (1, ))
    assert_size_stride(primals_628, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_629, (32, ), (1, ))
    assert_size_stride(primals_630, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_631, (32, ), (1, ))
    assert_size_stride(primals_632, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_633, (64, ), (1, ))
    assert_size_stride(primals_634, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_635, (32, ), (1, ))
    assert_size_stride(primals_636, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_637, (32, ), (1, ))
    assert_size_stride(primals_638, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_639, (32, ), (1, ))
    assert_size_stride(primals_640, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_641, (32, ), (1, ))
    assert_size_stride(primals_642, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_643, (64, ), (1, ))
    assert_size_stride(primals_644, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_645, (32, ), (1, ))
    assert_size_stride(primals_646, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_647, (32, ), (1, ))
    assert_size_stride(primals_648, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_649, (32, ), (1, ))
    assert_size_stride(primals_650, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_651, (32, ), (1, ))
    assert_size_stride(primals_652, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_653, (64, ), (1, ))
    assert_size_stride(primals_654, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_655, (32, ), (1, ))
    assert_size_stride(primals_656, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_657, (32, ), (1, ))
    assert_size_stride(primals_658, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_659, (32, ), (1, ))
    assert_size_stride(primals_660, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_661, (32, ), (1, ))
    assert_size_stride(primals_662, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_663, (64, ), (1, ))
    assert_size_stride(primals_664, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_665, (32, ), (1, ))
    assert_size_stride(primals_666, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_667, (32, ), (1, ))
    assert_size_stride(primals_668, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_669, (32, ), (1, ))
    assert_size_stride(primals_670, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_671, (32, ), (1, ))
    assert_size_stride(primals_672, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_673, (64, ), (1, ))
    assert_size_stride(primals_674, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_675, (32, ), (1, ))
    assert_size_stride(primals_676, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_677, (32, ), (1, ))
    assert_size_stride(primals_678, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_679, (32, ), (1, ))
    assert_size_stride(primals_680, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_681, (32, ), (1, ))
    assert_size_stride(primals_682, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_683, (64, ), (1, ))
    assert_size_stride(primals_684, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_685, (32, ), (1, ))
    assert_size_stride(primals_686, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_687, (32, ), (1, ))
    assert_size_stride(primals_688, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_689, (32, ), (1, ))
    assert_size_stride(primals_690, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_691, (32, ), (1, ))
    assert_size_stride(primals_692, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_693, (64, ), (1, ))
    assert_size_stride(primals_694, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_695, (64, ), (1, ))
    assert_size_stride(primals_696, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_697, (64, ), (1, ))
    assert_size_stride(primals_698, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_699, (64, ), (1, ))
    assert_size_stride(primals_700, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_701, (64, ), (1, ))
    assert_size_stride(primals_702, (3, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_703, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf693 = empty_strided_cuda((128, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_0.run(buf693, 128, grid=grid(128), stream=stream0)
        buf696 = empty_strided_cuda((256, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_1.run(buf696, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [fea], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [fea], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf1, primals_2, 1048576, grid=grid(1048576), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf980 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_1, x1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf2, primals_5, buf980, 524288, grid=grid(524288), stream=stream0)
        buf3 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf1, buf2, primals_5, buf3, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf979 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_2, x2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf4, primals_7, buf979, 524288, grid=grid(524288), stream=stream0)
        buf5 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf1, buf2, primals_5, buf4, primals_7, buf5, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf978 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_3, x3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf6, primals_9, buf978, 524288, grid=grid(524288), stream=stream0)
        buf7 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf1, buf2, primals_5, buf4, primals_7, buf6, primals_9, buf7, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf977 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_4, x4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf8, primals_11, buf977, 524288, grid=grid(524288), stream=stream0)
        buf9 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf1, buf2, primals_5, buf4, primals_7, buf6, primals_9, buf8, primals_11, buf9, 3145728, grid=grid(3145728), stream=stream0)
        del buf2
        del buf4
        del buf6
        del buf8
        del primals_11
        del primals_5
        del primals_7
        del primals_9
        # Topologically Sorted Source Nodes: [x5], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x5, mul, out], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf11, primals_13, buf1, 1048576, grid=grid(1048576), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf976 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_6, x1_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf12, primals_15, buf976, 524288, grid=grid(524288), stream=stream0)
        buf13 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf11, buf12, primals_15, buf13, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf975 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_7, x2_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf14, primals_17, buf975, 524288, grid=grid(524288), stream=stream0)
        buf15 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf11, buf12, primals_15, buf14, primals_17, buf15, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf974 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_8, x3_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf16, primals_19, buf974, 524288, grid=grid(524288), stream=stream0)
        buf17 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf11, buf12, primals_15, buf14, primals_17, buf16, primals_19, buf17, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf973 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_9, x4_1], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf18, primals_21, buf973, 524288, grid=grid(524288), stream=stream0)
        buf19 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf11, buf12, primals_15, buf14, primals_17, buf16, primals_19, buf18, primals_21, buf19, 3145728, grid=grid(3145728), stream=stream0)
        del buf12
        del buf14
        del buf16
        del buf18
        del primals_15
        del primals_17
        del primals_19
        del primals_21
        # Topologically Sorted Source Nodes: [x5_1], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x5_1, mul_1, out_1], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf21, primals_23, buf11, 1048576, grid=grid(1048576), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf972 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_11, x1_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf22, primals_25, buf972, 524288, grid=grid(524288), stream=stream0)
        buf23 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf21, buf22, primals_25, buf23, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf971 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_12, x2_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf24, primals_27, buf971, 524288, grid=grid(524288), stream=stream0)
        buf25 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf21, buf22, primals_25, buf24, primals_27, buf25, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf970 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_13, x3_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf26, primals_29, buf970, 524288, grid=grid(524288), stream=stream0)
        buf27 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf21, buf22, primals_25, buf24, primals_27, buf26, primals_29, buf27, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf969 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_14, x4_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf28, primals_31, buf969, 524288, grid=grid(524288), stream=stream0)
        buf29 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf21, buf22, primals_25, buf24, primals_27, buf26, primals_29, buf28, primals_31, buf29, 3145728, grid=grid(3145728), stream=stream0)
        del buf22
        del buf24
        del buf26
        del buf28
        del primals_25
        del primals_27
        del primals_29
        del primals_31
        # Topologically Sorted Source Nodes: [x5_2], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x5_2, mul_2, out_2, mul_3, input_1], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf31, primals_33, buf21, buf1, 1048576, grid=grid(1048576), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf968 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_16, x1_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf32, primals_35, buf968, 524288, grid=grid(524288), stream=stream0)
        buf33 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf31, buf32, primals_35, buf33, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf967 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_17, x2_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf34, primals_37, buf967, 524288, grid=grid(524288), stream=stream0)
        buf35 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf31, buf32, primals_35, buf34, primals_37, buf35, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf966 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_18, x3_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf36, primals_39, buf966, 524288, grid=grid(524288), stream=stream0)
        buf37 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf31, buf32, primals_35, buf34, primals_37, buf36, primals_39, buf37, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf965 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_19, x4_3], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf38, primals_41, buf965, 524288, grid=grid(524288), stream=stream0)
        buf39 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf31, buf32, primals_35, buf34, primals_37, buf36, primals_39, buf38, primals_41, buf39, 3145728, grid=grid(3145728), stream=stream0)
        del buf32
        del buf34
        del buf36
        del buf38
        del primals_35
        del primals_37
        del primals_39
        del primals_41
        # Topologically Sorted Source Nodes: [x5_3], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x5_3, mul_4, out_3], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf41, primals_43, buf31, 1048576, grid=grid(1048576), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf964 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_21, x1_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf42, primals_45, buf964, 524288, grid=grid(524288), stream=stream0)
        buf43 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf41, buf42, primals_45, buf43, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf963 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_22, x2_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf44, primals_47, buf963, 524288, grid=grid(524288), stream=stream0)
        buf45 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf41, buf42, primals_45, buf44, primals_47, buf45, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf962 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_23, x3_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf46, primals_49, buf962, 524288, grid=grid(524288), stream=stream0)
        buf47 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf41, buf42, primals_45, buf44, primals_47, buf46, primals_49, buf47, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf961 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_24, x4_4], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf48, primals_51, buf961, 524288, grid=grid(524288), stream=stream0)
        buf49 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_19], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf41, buf42, primals_45, buf44, primals_47, buf46, primals_49, buf48, primals_51, buf49, 3145728, grid=grid(3145728), stream=stream0)
        del buf42
        del buf44
        del buf46
        del buf48
        del primals_45
        del primals_47
        del primals_49
        del primals_51
        # Topologically Sorted Source Nodes: [x5_4], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x5_4, mul_5, out_4], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf51, primals_53, buf41, 1048576, grid=grid(1048576), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf960 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_26, x1_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf52, primals_55, buf960, 524288, grid=grid(524288), stream=stream0)
        buf53 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf51, buf52, primals_55, buf53, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf959 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_27, x2_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf54, primals_57, buf959, 524288, grid=grid(524288), stream=stream0)
        buf55 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf51, buf52, primals_55, buf54, primals_57, buf55, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf958 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_28, x3_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf56, primals_59, buf958, 524288, grid=grid(524288), stream=stream0)
        buf57 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_22], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf51, buf52, primals_55, buf54, primals_57, buf56, primals_59, buf57, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf957 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_29, x4_5], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf58, primals_61, buf957, 524288, grid=grid(524288), stream=stream0)
        buf59 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_23], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf51, buf52, primals_55, buf54, primals_57, buf56, primals_59, buf58, primals_61, buf59, 3145728, grid=grid(3145728), stream=stream0)
        del buf52
        del buf54
        del buf56
        del buf58
        del primals_55
        del primals_57
        del primals_59
        del primals_61
        # Topologically Sorted Source Nodes: [x5_5], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x5_5, mul_6, out_5, mul_7, input_2], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf61, primals_63, buf51, buf31, 1048576, grid=grid(1048576), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf956 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_31, x1_6], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf62, primals_65, buf956, 524288, grid=grid(524288), stream=stream0)
        buf63 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_24], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf61, buf62, primals_65, buf63, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf955 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_32, x2_6], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf64, primals_67, buf955, 524288, grid=grid(524288), stream=stream0)
        buf65 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_25], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf61, buf62, primals_65, buf64, primals_67, buf65, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf954 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_33, x3_6], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf66, primals_69, buf954, 524288, grid=grid(524288), stream=stream0)
        buf67 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_26], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf61, buf62, primals_65, buf64, primals_67, buf66, primals_69, buf67, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf953 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_34, x4_6], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf68, primals_71, buf953, 524288, grid=grid(524288), stream=stream0)
        buf69 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_27], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf61, buf62, primals_65, buf64, primals_67, buf66, primals_69, buf68, primals_71, buf69, 3145728, grid=grid(3145728), stream=stream0)
        del buf62
        del buf64
        del buf66
        del buf68
        del primals_65
        del primals_67
        del primals_69
        del primals_71
        # Topologically Sorted Source Nodes: [x5_6], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x5_6, mul_8, out_6], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf71, primals_73, buf61, 1048576, grid=grid(1048576), stream=stream0)
        del primals_73
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf952 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_36, x1_7], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf72, primals_75, buf952, 524288, grid=grid(524288), stream=stream0)
        buf73 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf71, buf72, primals_75, buf73, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf951 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_37, x2_7], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf74, primals_77, buf951, 524288, grid=grid(524288), stream=stream0)
        buf75 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_29], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf71, buf72, primals_75, buf74, primals_77, buf75, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf950 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_38, x3_7], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf76, primals_79, buf950, 524288, grid=grid(524288), stream=stream0)
        buf77 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf71, buf72, primals_75, buf74, primals_77, buf76, primals_79, buf77, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf949 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_39, x4_7], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf78, primals_81, buf949, 524288, grid=grid(524288), stream=stream0)
        buf79 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_31], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf71, buf72, primals_75, buf74, primals_77, buf76, primals_79, buf78, primals_81, buf79, 3145728, grid=grid(3145728), stream=stream0)
        del buf72
        del buf74
        del buf76
        del buf78
        del primals_75
        del primals_77
        del primals_79
        del primals_81
        # Topologically Sorted Source Nodes: [x5_7], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x5_7, mul_9, out_7], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf81, primals_83, buf71, 1048576, grid=grid(1048576), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf948 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_41, x1_8], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf82, primals_85, buf948, 524288, grid=grid(524288), stream=stream0)
        buf83 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_32], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf81, buf82, primals_85, buf83, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf947 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_42, x2_8], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf84, primals_87, buf947, 524288, grid=grid(524288), stream=stream0)
        buf85 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf81, buf82, primals_85, buf84, primals_87, buf85, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf946 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_43, x3_8], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf86, primals_89, buf946, 524288, grid=grid(524288), stream=stream0)
        buf87 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_34], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf81, buf82, primals_85, buf84, primals_87, buf86, primals_89, buf87, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_44], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf945 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_44, x4_8], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf88, primals_91, buf945, 524288, grid=grid(524288), stream=stream0)
        buf89 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_35], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf81, buf82, primals_85, buf84, primals_87, buf86, primals_89, buf88, primals_91, buf89, 3145728, grid=grid(3145728), stream=stream0)
        del buf82
        del buf84
        del buf86
        del buf88
        del primals_85
        del primals_87
        del primals_89
        del primals_91
        # Topologically Sorted Source Nodes: [x5_8], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x5_8, mul_10, out_8, mul_11, input_3], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf91, primals_93, buf81, buf61, 1048576, grid=grid(1048576), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [conv2d_46], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf944 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_46, x1_9], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf92, primals_95, buf944, 524288, grid=grid(524288), stream=stream0)
        buf93 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf91, buf92, primals_95, buf93, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf943 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_47, x2_9], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf94, primals_97, buf943, 524288, grid=grid(524288), stream=stream0)
        buf95 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf91, buf92, primals_95, buf94, primals_97, buf95, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf942 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_48, x3_9], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf96, primals_99, buf942, 524288, grid=grid(524288), stream=stream0)
        buf97 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf91, buf92, primals_95, buf94, primals_97, buf96, primals_99, buf97, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf941 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_49, x4_9], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf98, primals_101, buf941, 524288, grid=grid(524288), stream=stream0)
        buf99 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf91, buf92, primals_95, buf94, primals_97, buf96, primals_99, buf98, primals_101, buf99, 3145728, grid=grid(3145728), stream=stream0)
        del buf92
        del buf94
        del buf96
        del buf98
        del primals_101
        del primals_95
        del primals_97
        del primals_99
        # Topologically Sorted Source Nodes: [x5_9], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x5_9, mul_12, out_9], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf101, primals_103, buf91, 1048576, grid=grid(1048576), stream=stream0)
        del primals_103
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf940 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_51, x1_10], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf102, primals_105, buf940, 524288, grid=grid(524288), stream=stream0)
        buf103 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf101, buf102, primals_105, buf103, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf939 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_52, x2_10], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf104, primals_107, buf939, 524288, grid=grid(524288), stream=stream0)
        buf105 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf101, buf102, primals_105, buf104, primals_107, buf105, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf938 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_53, x3_10], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf106, primals_109, buf938, 524288, grid=grid(524288), stream=stream0)
        buf107 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_42], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf101, buf102, primals_105, buf104, primals_107, buf106, primals_109, buf107, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_54], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf937 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_54, x4_10], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf108, primals_111, buf937, 524288, grid=grid(524288), stream=stream0)
        buf109 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_43], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf101, buf102, primals_105, buf104, primals_107, buf106, primals_109, buf108, primals_111, buf109, 3145728, grid=grid(3145728), stream=stream0)
        del buf102
        del buf104
        del buf106
        del buf108
        del primals_105
        del primals_107
        del primals_109
        del primals_111
        # Topologically Sorted Source Nodes: [x5_10], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x5_10, mul_13, out_10], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf111, primals_113, buf101, 1048576, grid=grid(1048576), stream=stream0)
        del primals_113
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf936 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_56, x1_11], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf112, primals_115, buf936, 524288, grid=grid(524288), stream=stream0)
        buf113 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_44], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf111, buf112, primals_115, buf113, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf935 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_57, x2_11], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf114, primals_117, buf935, 524288, grid=grid(524288), stream=stream0)
        buf115 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_45], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf111, buf112, primals_115, buf114, primals_117, buf115, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf934 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_58, x3_11], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf116, primals_119, buf934, 524288, grid=grid(524288), stream=stream0)
        buf117 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_46], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf111, buf112, primals_115, buf114, primals_117, buf116, primals_119, buf117, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf933 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_59, x4_11], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf118, primals_121, buf933, 524288, grid=grid(524288), stream=stream0)
        buf119 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_47], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf111, buf112, primals_115, buf114, primals_117, buf116, primals_119, buf118, primals_121, buf119, 3145728, grid=grid(3145728), stream=stream0)
        del buf112
        del buf114
        del buf116
        del buf118
        del primals_115
        del primals_117
        del primals_119
        del primals_121
        # Topologically Sorted Source Nodes: [x5_11], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x5_11, mul_14, out_11, mul_15, input_4], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf121, primals_123, buf111, buf91, 1048576, grid=grid(1048576), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf932 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_61, x1_12], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf122, primals_125, buf932, 524288, grid=grid(524288), stream=stream0)
        buf123 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_48], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf121, buf122, primals_125, buf123, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf931 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_62, x2_12], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf124, primals_127, buf931, 524288, grid=grid(524288), stream=stream0)
        buf125 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_49], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf121, buf122, primals_125, buf124, primals_127, buf125, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf930 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_63, x3_12], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf126, primals_129, buf930, 524288, grid=grid(524288), stream=stream0)
        buf127 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_50], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf121, buf122, primals_125, buf124, primals_127, buf126, primals_129, buf127, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf929 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_64, x4_12], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf128, primals_131, buf929, 524288, grid=grid(524288), stream=stream0)
        buf129 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_51], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf121, buf122, primals_125, buf124, primals_127, buf126, primals_129, buf128, primals_131, buf129, 3145728, grid=grid(3145728), stream=stream0)
        del buf122
        del buf124
        del buf126
        del buf128
        del primals_125
        del primals_127
        del primals_129
        del primals_131
        # Topologically Sorted Source Nodes: [x5_12], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x5_12, mul_16, out_12], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf131, primals_133, buf121, 1048576, grid=grid(1048576), stream=stream0)
        del primals_133
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf928 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_66, x1_13], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf132, primals_135, buf928, 524288, grid=grid(524288), stream=stream0)
        buf133 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_52], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf131, buf132, primals_135, buf133, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_67], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf927 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_67, x2_13], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf134, primals_137, buf927, 524288, grid=grid(524288), stream=stream0)
        buf135 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_53], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf131, buf132, primals_135, buf134, primals_137, buf135, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_68], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf926 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_68, x3_13], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf136, primals_139, buf926, 524288, grid=grid(524288), stream=stream0)
        buf137 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf131, buf132, primals_135, buf134, primals_137, buf136, primals_139, buf137, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_69], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf925 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_69, x4_13], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf138, primals_141, buf925, 524288, grid=grid(524288), stream=stream0)
        buf139 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_55], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf131, buf132, primals_135, buf134, primals_137, buf136, primals_139, buf138, primals_141, buf139, 3145728, grid=grid(3145728), stream=stream0)
        del buf132
        del buf134
        del buf136
        del buf138
        del primals_135
        del primals_137
        del primals_139
        del primals_141
        # Topologically Sorted Source Nodes: [x5_13], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x5_13, mul_17, out_13], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf141, primals_143, buf131, 1048576, grid=grid(1048576), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [conv2d_71], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf924 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_71, x1_14], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf142, primals_145, buf924, 524288, grid=grid(524288), stream=stream0)
        buf143 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_56], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf141, buf142, primals_145, buf143, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_72], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf923 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_72, x2_14], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf144, primals_147, buf923, 524288, grid=grid(524288), stream=stream0)
        buf145 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_57], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf141, buf142, primals_145, buf144, primals_147, buf145, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_73], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf922 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_73, x3_14], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf146, primals_149, buf922, 524288, grid=grid(524288), stream=stream0)
        buf147 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_58], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf141, buf142, primals_145, buf144, primals_147, buf146, primals_149, buf147, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_74], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf921 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_74, x4_14], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf148, primals_151, buf921, 524288, grid=grid(524288), stream=stream0)
        buf149 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_59], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf141, buf142, primals_145, buf144, primals_147, buf146, primals_149, buf148, primals_151, buf149, 3145728, grid=grid(3145728), stream=stream0)
        del buf142
        del buf144
        del buf146
        del buf148
        del primals_145
        del primals_147
        del primals_149
        del primals_151
        # Topologically Sorted Source Nodes: [x5_14], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x5_14, mul_18, out_14, mul_19, input_5], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf151, primals_153, buf141, buf121, 1048576, grid=grid(1048576), stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [conv2d_76], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf920 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_76, x1_15], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf152, primals_155, buf920, 524288, grid=grid(524288), stream=stream0)
        buf153 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_60], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf151, buf152, primals_155, buf153, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_77], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf919 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_77, x2_15], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf154, primals_157, buf919, 524288, grid=grid(524288), stream=stream0)
        buf155 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_61], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf151, buf152, primals_155, buf154, primals_157, buf155, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_78], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_158, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf918 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_78, x3_15], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf156, primals_159, buf918, 524288, grid=grid(524288), stream=stream0)
        buf157 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_62], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf151, buf152, primals_155, buf154, primals_157, buf156, primals_159, buf157, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_79], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf917 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_79, x4_15], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf158, primals_161, buf917, 524288, grid=grid(524288), stream=stream0)
        buf159 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_63], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf151, buf152, primals_155, buf154, primals_157, buf156, primals_159, buf158, primals_161, buf159, 3145728, grid=grid(3145728), stream=stream0)
        del buf152
        del buf154
        del buf156
        del buf158
        del primals_155
        del primals_157
        del primals_159
        del primals_161
        # Topologically Sorted Source Nodes: [x5_15], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x5_15, mul_20, out_15], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf161, primals_163, buf151, 1048576, grid=grid(1048576), stream=stream0)
        del primals_163
        # Topologically Sorted Source Nodes: [conv2d_81], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf916 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_81, x1_16], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf162, primals_165, buf916, 524288, grid=grid(524288), stream=stream0)
        buf163 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_64], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf161, buf162, primals_165, buf163, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_82], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_166, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf915 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_82, x2_16], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf164, primals_167, buf915, 524288, grid=grid(524288), stream=stream0)
        buf165 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_65], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf161, buf162, primals_165, buf164, primals_167, buf165, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_83], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_168, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf914 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_83, x3_16], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf166, primals_169, buf914, 524288, grid=grid(524288), stream=stream0)
        buf167 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_66], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf161, buf162, primals_165, buf164, primals_167, buf166, primals_169, buf167, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_84], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf913 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_84, x4_16], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf168, primals_171, buf913, 524288, grid=grid(524288), stream=stream0)
        buf169 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_67], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf161, buf162, primals_165, buf164, primals_167, buf166, primals_169, buf168, primals_171, buf169, 3145728, grid=grid(3145728), stream=stream0)
        del buf162
        del buf164
        del buf166
        del buf168
        del primals_165
        del primals_167
        del primals_169
        del primals_171
        # Topologically Sorted Source Nodes: [x5_16], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [x5_16, mul_21, out_16], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf171, primals_173, buf161, 1048576, grid=grid(1048576), stream=stream0)
        del primals_173
        # Topologically Sorted Source Nodes: [conv2d_86], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf912 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_86, x1_17], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf172, primals_175, buf912, 524288, grid=grid(524288), stream=stream0)
        buf173 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_68], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf171, buf172, primals_175, buf173, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_87], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf911 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_87, x2_17], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf174, primals_177, buf911, 524288, grid=grid(524288), stream=stream0)
        buf175 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_69], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf171, buf172, primals_175, buf174, primals_177, buf175, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_88], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_178, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf910 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_88, x3_17], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf176, primals_179, buf910, 524288, grid=grid(524288), stream=stream0)
        buf177 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_70], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf171, buf172, primals_175, buf174, primals_177, buf176, primals_179, buf177, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_89], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf909 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_89, x4_17], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf178, primals_181, buf909, 524288, grid=grid(524288), stream=stream0)
        buf179 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_71], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf171, buf172, primals_175, buf174, primals_177, buf176, primals_179, buf178, primals_181, buf179, 3145728, grid=grid(3145728), stream=stream0)
        del buf172
        del buf174
        del buf176
        del buf178
        del primals_175
        del primals_177
        del primals_179
        del primals_181
        # Topologically Sorted Source Nodes: [x5_17], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x5_17, mul_22, out_17, mul_23, input_6], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf181, primals_183, buf171, buf151, 1048576, grid=grid(1048576), stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [conv2d_91], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_184, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf908 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_91, x1_18], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf182, primals_185, buf908, 524288, grid=grid(524288), stream=stream0)
        buf183 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_72], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf181, buf182, primals_185, buf183, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_92], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf907 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_92, x2_18], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf184, primals_187, buf907, 524288, grid=grid(524288), stream=stream0)
        buf185 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_73], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf181, buf182, primals_185, buf184, primals_187, buf185, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_93], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_188, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf906 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_93, x3_18], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf186, primals_189, buf906, 524288, grid=grid(524288), stream=stream0)
        buf187 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_74], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf181, buf182, primals_185, buf184, primals_187, buf186, primals_189, buf187, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_94], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf905 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_94, x4_18], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf188, primals_191, buf905, 524288, grid=grid(524288), stream=stream0)
        buf189 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_75], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf181, buf182, primals_185, buf184, primals_187, buf186, primals_189, buf188, primals_191, buf189, 3145728, grid=grid(3145728), stream=stream0)
        del buf182
        del buf184
        del buf186
        del buf188
        del primals_185
        del primals_187
        del primals_189
        del primals_191
        # Topologically Sorted Source Nodes: [x5_18], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf191 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x5_18, mul_24, out_18], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf191, primals_193, buf181, 1048576, grid=grid(1048576), stream=stream0)
        del primals_193
        # Topologically Sorted Source Nodes: [conv2d_96], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_194, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf904 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_96, x1_19], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf192, primals_195, buf904, 524288, grid=grid(524288), stream=stream0)
        buf193 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_76], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf191, buf192, primals_195, buf193, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_97], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf903 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_97, x2_19], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf194, primals_197, buf903, 524288, grid=grid(524288), stream=stream0)
        buf195 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_77], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf191, buf192, primals_195, buf194, primals_197, buf195, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_98], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_198, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf902 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_98, x3_19], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf196, primals_199, buf902, 524288, grid=grid(524288), stream=stream0)
        buf197 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_78], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf191, buf192, primals_195, buf194, primals_197, buf196, primals_199, buf197, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_99], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf901 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_99, x4_19], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf198, primals_201, buf901, 524288, grid=grid(524288), stream=stream0)
        buf199 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_79], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf191, buf192, primals_195, buf194, primals_197, buf196, primals_199, buf198, primals_201, buf199, 3145728, grid=grid(3145728), stream=stream0)
        del buf192
        del buf194
        del buf196
        del buf198
        del primals_195
        del primals_197
        del primals_199
        del primals_201
        # Topologically Sorted Source Nodes: [x5_19], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [x5_19, mul_25, out_19], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf201, primals_203, buf191, 1048576, grid=grid(1048576), stream=stream0)
        del primals_203
        # Topologically Sorted Source Nodes: [conv2d_101], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_204, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf900 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_101, x1_20], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf202, primals_205, buf900, 524288, grid=grid(524288), stream=stream0)
        buf203 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_80], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf201, buf202, primals_205, buf203, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_102], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf899 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_102, x2_20], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf204, primals_207, buf899, 524288, grid=grid(524288), stream=stream0)
        buf205 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_81], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf201, buf202, primals_205, buf204, primals_207, buf205, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_103], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf898 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_103, x3_20], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf206, primals_209, buf898, 524288, grid=grid(524288), stream=stream0)
        buf207 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_82], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf201, buf202, primals_205, buf204, primals_207, buf206, primals_209, buf207, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_104], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_210, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf897 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_104, x4_20], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf208, primals_211, buf897, 524288, grid=grid(524288), stream=stream0)
        buf209 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_83], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf201, buf202, primals_205, buf204, primals_207, buf206, primals_209, buf208, primals_211, buf209, 3145728, grid=grid(3145728), stream=stream0)
        del buf202
        del buf204
        del buf206
        del buf208
        del primals_205
        del primals_207
        del primals_209
        del primals_211
        # Topologically Sorted Source Nodes: [x5_20], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [x5_20, mul_26, out_20, mul_27, input_7], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf211, primals_213, buf201, buf181, 1048576, grid=grid(1048576), stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [conv2d_106], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf896 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_106, x1_21], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf212, primals_215, buf896, 524288, grid=grid(524288), stream=stream0)
        buf213 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_84], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf211, buf212, primals_215, buf213, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_107], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf895 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_107, x2_21], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf214, primals_217, buf895, 524288, grid=grid(524288), stream=stream0)
        buf215 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_85], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf211, buf212, primals_215, buf214, primals_217, buf215, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_108], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf894 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_108, x3_21], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf216, primals_219, buf894, 524288, grid=grid(524288), stream=stream0)
        buf217 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_86], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf211, buf212, primals_215, buf214, primals_217, buf216, primals_219, buf217, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_109], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_220, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf893 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_109, x4_21], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf218, primals_221, buf893, 524288, grid=grid(524288), stream=stream0)
        buf219 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_87], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf211, buf212, primals_215, buf214, primals_217, buf216, primals_219, buf218, primals_221, buf219, 3145728, grid=grid(3145728), stream=stream0)
        del buf212
        del buf214
        del buf216
        del buf218
        del primals_215
        del primals_217
        del primals_219
        del primals_221
        # Topologically Sorted Source Nodes: [x5_21], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf221 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [x5_21, mul_28, out_21], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf221, primals_223, buf211, 1048576, grid=grid(1048576), stream=stream0)
        del primals_223
        # Topologically Sorted Source Nodes: [conv2d_111], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf892 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_111, x1_22], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf222, primals_225, buf892, 524288, grid=grid(524288), stream=stream0)
        buf223 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_88], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf221, buf222, primals_225, buf223, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_112], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf891 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_112, x2_22], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf224, primals_227, buf891, 524288, grid=grid(524288), stream=stream0)
        buf225 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_89], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf221, buf222, primals_225, buf224, primals_227, buf225, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_113], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_228, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf890 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_113, x3_22], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf226, primals_229, buf890, 524288, grid=grid(524288), stream=stream0)
        buf227 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_90], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf221, buf222, primals_225, buf224, primals_227, buf226, primals_229, buf227, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_114], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf889 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_114, x4_22], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf228, primals_231, buf889, 524288, grid=grid(524288), stream=stream0)
        buf229 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_91], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf221, buf222, primals_225, buf224, primals_227, buf226, primals_229, buf228, primals_231, buf229, 3145728, grid=grid(3145728), stream=stream0)
        del buf222
        del buf224
        del buf226
        del buf228
        del primals_225
        del primals_227
        del primals_229
        del primals_231
        # Topologically Sorted Source Nodes: [x5_22], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x5_22, mul_29, out_22], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf231, primals_233, buf221, 1048576, grid=grid(1048576), stream=stream0)
        del primals_233
        # Topologically Sorted Source Nodes: [conv2d_116], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_234, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf888 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_116, x1_23], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf232, primals_235, buf888, 524288, grid=grid(524288), stream=stream0)
        buf233 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_92], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf231, buf232, primals_235, buf233, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_117], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_236, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf887 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_117, x2_23], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf234, primals_237, buf887, 524288, grid=grid(524288), stream=stream0)
        buf235 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_93], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf231, buf232, primals_235, buf234, primals_237, buf235, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_118], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf886 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_118, x3_23], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf236, primals_239, buf886, 524288, grid=grid(524288), stream=stream0)
        buf237 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_94], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf231, buf232, primals_235, buf234, primals_237, buf236, primals_239, buf237, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_119], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_240, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf885 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_119, x4_23], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf238, primals_241, buf885, 524288, grid=grid(524288), stream=stream0)
        buf239 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_95], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf231, buf232, primals_235, buf234, primals_237, buf236, primals_239, buf238, primals_241, buf239, 3145728, grid=grid(3145728), stream=stream0)
        del buf232
        del buf234
        del buf236
        del buf238
        del primals_235
        del primals_237
        del primals_239
        del primals_241
        # Topologically Sorted Source Nodes: [x5_23], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [x5_23, mul_30, out_23, mul_31, input_8], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf241, primals_243, buf231, buf211, 1048576, grid=grid(1048576), stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [conv2d_121], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf884 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_121, x1_24], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf242, primals_245, buf884, 524288, grid=grid(524288), stream=stream0)
        buf243 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_96], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf241, buf242, primals_245, buf243, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_122], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_246, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf883 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_122, x2_24], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf244, primals_247, buf883, 524288, grid=grid(524288), stream=stream0)
        buf245 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_97], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf241, buf242, primals_245, buf244, primals_247, buf245, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_123], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_248, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf882 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_123, x3_24], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf246, primals_249, buf882, 524288, grid=grid(524288), stream=stream0)
        buf247 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_98], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf241, buf242, primals_245, buf244, primals_247, buf246, primals_249, buf247, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_124], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf881 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_124, x4_24], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf248, primals_251, buf881, 524288, grid=grid(524288), stream=stream0)
        buf249 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_99], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf241, buf242, primals_245, buf244, primals_247, buf246, primals_249, buf248, primals_251, buf249, 3145728, grid=grid(3145728), stream=stream0)
        del buf242
        del buf244
        del buf246
        del buf248
        del primals_245
        del primals_247
        del primals_249
        del primals_251
        # Topologically Sorted Source Nodes: [x5_24], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf251 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [x5_24, mul_32, out_24], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf251, primals_253, buf241, 1048576, grid=grid(1048576), stream=stream0)
        del primals_253
        # Topologically Sorted Source Nodes: [conv2d_126], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_254, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf880 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_126, x1_25], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf252, primals_255, buf880, 524288, grid=grid(524288), stream=stream0)
        buf253 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_100], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf251, buf252, primals_255, buf253, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_127], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf879 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_127, x2_25], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf254, primals_257, buf879, 524288, grid=grid(524288), stream=stream0)
        buf255 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_101], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf251, buf252, primals_255, buf254, primals_257, buf255, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_128], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_258, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf878 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_128, x3_25], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf256, primals_259, buf878, 524288, grid=grid(524288), stream=stream0)
        buf257 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_102], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf251, buf252, primals_255, buf254, primals_257, buf256, primals_259, buf257, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_129], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_260, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf877 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_129, x4_25], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf258, primals_261, buf877, 524288, grid=grid(524288), stream=stream0)
        buf259 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_103], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf251, buf252, primals_255, buf254, primals_257, buf256, primals_259, buf258, primals_261, buf259, 3145728, grid=grid(3145728), stream=stream0)
        del buf252
        del buf254
        del buf256
        del buf258
        del primals_255
        del primals_257
        del primals_259
        del primals_261
        # Topologically Sorted Source Nodes: [x5_25], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf261 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x5_25, mul_33, out_25], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf261, primals_263, buf251, 1048576, grid=grid(1048576), stream=stream0)
        del primals_263
        # Topologically Sorted Source Nodes: [conv2d_131], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_264, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf876 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_131, x1_26], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf262, primals_265, buf876, 524288, grid=grid(524288), stream=stream0)
        buf263 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_104], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf261, buf262, primals_265, buf263, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_132], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_266, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf875 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_132, x2_26], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf264, primals_267, buf875, 524288, grid=grid(524288), stream=stream0)
        buf265 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_105], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf261, buf262, primals_265, buf264, primals_267, buf265, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_133], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf874 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_133, x3_26], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf266, primals_269, buf874, 524288, grid=grid(524288), stream=stream0)
        buf267 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_106], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf261, buf262, primals_265, buf264, primals_267, buf266, primals_269, buf267, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_134], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_270, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf873 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_134, x4_26], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf268, primals_271, buf873, 524288, grid=grid(524288), stream=stream0)
        buf269 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_107], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf261, buf262, primals_265, buf264, primals_267, buf266, primals_269, buf268, primals_271, buf269, 3145728, grid=grid(3145728), stream=stream0)
        del buf262
        del buf264
        del buf266
        del buf268
        del primals_265
        del primals_267
        del primals_269
        del primals_271
        # Topologically Sorted Source Nodes: [x5_26], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [x5_26, mul_34, out_26, mul_35, input_9], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf271, primals_273, buf261, buf241, 1048576, grid=grid(1048576), stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [conv2d_136], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_274, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf872 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_136, x1_27], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf272, primals_275, buf872, 524288, grid=grid(524288), stream=stream0)
        buf273 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_108], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf271, buf272, primals_275, buf273, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_137], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf871 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_137, x2_27], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf274, primals_277, buf871, 524288, grid=grid(524288), stream=stream0)
        buf275 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_109], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf271, buf272, primals_275, buf274, primals_277, buf275, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_138], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_278, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf870 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_138, x3_27], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf276, primals_279, buf870, 524288, grid=grid(524288), stream=stream0)
        buf277 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_110], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf271, buf272, primals_275, buf274, primals_277, buf276, primals_279, buf277, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_139], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_280, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf869 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_139, x4_27], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf278, primals_281, buf869, 524288, grid=grid(524288), stream=stream0)
        buf279 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_111], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf271, buf272, primals_275, buf274, primals_277, buf276, primals_279, buf278, primals_281, buf279, 3145728, grid=grid(3145728), stream=stream0)
        del buf272
        del buf274
        del buf276
        del buf278
        del primals_275
        del primals_277
        del primals_279
        del primals_281
        # Topologically Sorted Source Nodes: [x5_27], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf281 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [x5_27, mul_36, out_27], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf281, primals_283, buf271, 1048576, grid=grid(1048576), stream=stream0)
        del primals_283
        # Topologically Sorted Source Nodes: [conv2d_141], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_284, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf868 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_141, x1_28], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf282, primals_285, buf868, 524288, grid=grid(524288), stream=stream0)
        buf283 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_112], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf281, buf282, primals_285, buf283, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_142], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_286, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf867 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_142, x2_28], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf284, primals_287, buf867, 524288, grid=grid(524288), stream=stream0)
        buf285 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_113], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf281, buf282, primals_285, buf284, primals_287, buf285, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_143], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf866 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_143, x3_28], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf286, primals_289, buf866, 524288, grid=grid(524288), stream=stream0)
        buf287 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_114], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf281, buf282, primals_285, buf284, primals_287, buf286, primals_289, buf287, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_144], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_290, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf865 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_144, x4_28], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf288, primals_291, buf865, 524288, grid=grid(524288), stream=stream0)
        buf289 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_115], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf281, buf282, primals_285, buf284, primals_287, buf286, primals_289, buf288, primals_291, buf289, 3145728, grid=grid(3145728), stream=stream0)
        del buf282
        del buf284
        del buf286
        del buf288
        del primals_285
        del primals_287
        del primals_289
        del primals_291
        # Topologically Sorted Source Nodes: [x5_28], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf291 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [x5_28, mul_37, out_28], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf291, primals_293, buf281, 1048576, grid=grid(1048576), stream=stream0)
        del primals_293
        # Topologically Sorted Source Nodes: [conv2d_146], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf864 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_146, x1_29], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf292, primals_295, buf864, 524288, grid=grid(524288), stream=stream0)
        buf293 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_116], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf291, buf292, primals_295, buf293, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_147], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_296, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf863 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_147, x2_29], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf294, primals_297, buf863, 524288, grid=grid(524288), stream=stream0)
        buf295 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_117], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf291, buf292, primals_295, buf294, primals_297, buf295, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_148], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_298, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf862 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_148, x3_29], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf296, primals_299, buf862, 524288, grid=grid(524288), stream=stream0)
        buf297 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_118], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf291, buf292, primals_295, buf294, primals_297, buf296, primals_299, buf297, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_149], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_300, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf861 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_149, x4_29], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf298, primals_301, buf861, 524288, grid=grid(524288), stream=stream0)
        buf299 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_119], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf291, buf292, primals_295, buf294, primals_297, buf296, primals_299, buf298, primals_301, buf299, 3145728, grid=grid(3145728), stream=stream0)
        del buf292
        del buf294
        del buf296
        del buf298
        del primals_295
        del primals_297
        del primals_299
        del primals_301
        # Topologically Sorted Source Nodes: [x5_29], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf301 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [x5_29, mul_38, out_29, mul_39, input_10], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf301, primals_303, buf291, buf271, 1048576, grid=grid(1048576), stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [conv2d_151], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_304, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf860 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_151, x1_30], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf302, primals_305, buf860, 524288, grid=grid(524288), stream=stream0)
        buf303 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_120], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf301, buf302, primals_305, buf303, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_152], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf859 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_152, x2_30], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf304, primals_307, buf859, 524288, grid=grid(524288), stream=stream0)
        buf305 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_121], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf301, buf302, primals_305, buf304, primals_307, buf305, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_153], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_308, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf858 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_153, x3_30], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf306, primals_309, buf858, 524288, grid=grid(524288), stream=stream0)
        buf307 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_122], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf301, buf302, primals_305, buf304, primals_307, buf306, primals_309, buf307, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_154], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_310, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf857 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_154, x4_30], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf308, primals_311, buf857, 524288, grid=grid(524288), stream=stream0)
        buf309 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_123], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf301, buf302, primals_305, buf304, primals_307, buf306, primals_309, buf308, primals_311, buf309, 3145728, grid=grid(3145728), stream=stream0)
        del buf302
        del buf304
        del buf306
        del buf308
        del primals_305
        del primals_307
        del primals_309
        del primals_311
        # Topologically Sorted Source Nodes: [x5_30], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf311 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [x5_30, mul_40, out_30], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf311, primals_313, buf301, 1048576, grid=grid(1048576), stream=stream0)
        del primals_313
        # Topologically Sorted Source Nodes: [conv2d_156], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf856 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_156, x1_31], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf312, primals_315, buf856, 524288, grid=grid(524288), stream=stream0)
        buf313 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_124], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf311, buf312, primals_315, buf313, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_157], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_316, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf855 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_157, x2_31], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf314, primals_317, buf855, 524288, grid=grid(524288), stream=stream0)
        buf315 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_125], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf311, buf312, primals_315, buf314, primals_317, buf315, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_158], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf854 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_158, x3_31], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf316, primals_319, buf854, 524288, grid=grid(524288), stream=stream0)
        buf317 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_126], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf311, buf312, primals_315, buf314, primals_317, buf316, primals_319, buf317, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_159], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, primals_320, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf853 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_159, x4_31], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf318, primals_321, buf853, 524288, grid=grid(524288), stream=stream0)
        buf319 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_127], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf311, buf312, primals_315, buf314, primals_317, buf316, primals_319, buf318, primals_321, buf319, 3145728, grid=grid(3145728), stream=stream0)
        del buf312
        del buf314
        del buf316
        del buf318
        del primals_315
        del primals_317
        del primals_319
        del primals_321
        # Topologically Sorted Source Nodes: [x5_31], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [x5_31, mul_41, out_31], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf321, primals_323, buf311, 1048576, grid=grid(1048576), stream=stream0)
        del primals_323
        # Topologically Sorted Source Nodes: [conv2d_161], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_324, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf852 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_161, x1_32], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf322, primals_325, buf852, 524288, grid=grid(524288), stream=stream0)
        buf323 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_128], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf321, buf322, primals_325, buf323, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_162], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_326, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf851 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_162, x2_32], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf324, primals_327, buf851, 524288, grid=grid(524288), stream=stream0)
        buf325 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_129], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf321, buf322, primals_325, buf324, primals_327, buf325, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_163], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_328, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf850 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_163, x3_32], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf326, primals_329, buf850, 524288, grid=grid(524288), stream=stream0)
        buf327 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_130], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf321, buf322, primals_325, buf324, primals_327, buf326, primals_329, buf327, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_164], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_330, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf849 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_164, x4_32], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf328, primals_331, buf849, 524288, grid=grid(524288), stream=stream0)
        buf329 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_131], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf321, buf322, primals_325, buf324, primals_327, buf326, primals_329, buf328, primals_331, buf329, 3145728, grid=grid(3145728), stream=stream0)
        del buf322
        del buf324
        del buf326
        del buf328
        del primals_325
        del primals_327
        del primals_329
        del primals_331
        # Topologically Sorted Source Nodes: [x5_32], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf331 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [x5_32, mul_42, out_32, mul_43, input_11], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf331, primals_333, buf321, buf301, 1048576, grid=grid(1048576), stream=stream0)
        del primals_333
        # Topologically Sorted Source Nodes: [conv2d_166], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_334, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf848 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_166, x1_33], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf332, primals_335, buf848, 524288, grid=grid(524288), stream=stream0)
        buf333 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_132], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf331, buf332, primals_335, buf333, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_167], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_336, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf847 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_167, x2_33], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf334, primals_337, buf847, 524288, grid=grid(524288), stream=stream0)
        buf335 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_133], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf331, buf332, primals_335, buf334, primals_337, buf335, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_168], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_338, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf846 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_168, x3_33], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf336, primals_339, buf846, 524288, grid=grid(524288), stream=stream0)
        buf337 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_134], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf331, buf332, primals_335, buf334, primals_337, buf336, primals_339, buf337, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_169], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, primals_340, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf845 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_169, x4_33], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf338, primals_341, buf845, 524288, grid=grid(524288), stream=stream0)
        buf339 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_135], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf331, buf332, primals_335, buf334, primals_337, buf336, primals_339, buf338, primals_341, buf339, 3145728, grid=grid(3145728), stream=stream0)
        del buf332
        del buf334
        del buf336
        del buf338
        del primals_335
        del primals_337
        del primals_339
        del primals_341
        # Topologically Sorted Source Nodes: [x5_33], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf341 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [x5_33, mul_44, out_33], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf341, primals_343, buf331, 1048576, grid=grid(1048576), stream=stream0)
        del primals_343
        # Topologically Sorted Source Nodes: [conv2d_171], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf844 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_171, x1_34], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf342, primals_345, buf844, 524288, grid=grid(524288), stream=stream0)
        buf343 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_136], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf341, buf342, primals_345, buf343, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_172], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_346, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf843 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_172, x2_34], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf344, primals_347, buf843, 524288, grid=grid(524288), stream=stream0)
        buf345 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_137], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf341, buf342, primals_345, buf344, primals_347, buf345, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_173], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_348, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf842 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_173, x3_34], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf346, primals_349, buf842, 524288, grid=grid(524288), stream=stream0)
        buf347 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_138], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf341, buf342, primals_345, buf344, primals_347, buf346, primals_349, buf347, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_174], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, primals_350, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf841 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_174, x4_34], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf348, primals_351, buf841, 524288, grid=grid(524288), stream=stream0)
        buf349 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_139], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf341, buf342, primals_345, buf344, primals_347, buf346, primals_349, buf348, primals_351, buf349, 3145728, grid=grid(3145728), stream=stream0)
        del buf342
        del buf344
        del buf346
        del buf348
        del primals_345
        del primals_347
        del primals_349
        del primals_351
        # Topologically Sorted Source Nodes: [x5_34], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf351 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [x5_34, mul_45, out_34], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf351, primals_353, buf341, 1048576, grid=grid(1048576), stream=stream0)
        del primals_353
        # Topologically Sorted Source Nodes: [conv2d_176], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_354, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf840 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_176, x1_35], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf352, primals_355, buf840, 524288, grid=grid(524288), stream=stream0)
        buf353 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_140], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf351, buf352, primals_355, buf353, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_177], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_356, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf839 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_177, x2_35], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf354, primals_357, buf839, 524288, grid=grid(524288), stream=stream0)
        buf355 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_141], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf351, buf352, primals_355, buf354, primals_357, buf355, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_178], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_358, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf838 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_178, x3_35], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf356, primals_359, buf838, 524288, grid=grid(524288), stream=stream0)
        buf357 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_142], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf351, buf352, primals_355, buf354, primals_357, buf356, primals_359, buf357, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_179], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf837 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_179, x4_35], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf358, primals_361, buf837, 524288, grid=grid(524288), stream=stream0)
        buf359 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_143], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf351, buf352, primals_355, buf354, primals_357, buf356, primals_359, buf358, primals_361, buf359, 3145728, grid=grid(3145728), stream=stream0)
        del buf352
        del buf354
        del buf356
        del buf358
        del primals_355
        del primals_357
        del primals_359
        del primals_361
        # Topologically Sorted Source Nodes: [x5_35], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [x5_35, mul_46, out_35, mul_47, input_12], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf361, primals_363, buf351, buf331, 1048576, grid=grid(1048576), stream=stream0)
        del primals_363
        # Topologically Sorted Source Nodes: [conv2d_181], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_364, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf836 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_181, x1_36], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf362, primals_365, buf836, 524288, grid=grid(524288), stream=stream0)
        buf363 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_144], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf361, buf362, primals_365, buf363, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_182], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_366, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf835 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_182, x2_36], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf364, primals_367, buf835, 524288, grid=grid(524288), stream=stream0)
        buf365 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_145], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf361, buf362, primals_365, buf364, primals_367, buf365, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_183], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_368, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf834 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_183, x3_36], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf366, primals_369, buf834, 524288, grid=grid(524288), stream=stream0)
        buf367 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_146], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf361, buf362, primals_365, buf364, primals_367, buf366, primals_369, buf367, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_184], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, primals_370, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf833 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_184, x4_36], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf368, primals_371, buf833, 524288, grid=grid(524288), stream=stream0)
        buf369 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_147], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf361, buf362, primals_365, buf364, primals_367, buf366, primals_369, buf368, primals_371, buf369, 3145728, grid=grid(3145728), stream=stream0)
        del buf362
        del buf364
        del buf366
        del buf368
        del primals_365
        del primals_367
        del primals_369
        del primals_371
        # Topologically Sorted Source Nodes: [x5_36], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf369, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf371 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [x5_36, mul_48, out_36], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf371, primals_373, buf361, 1048576, grid=grid(1048576), stream=stream0)
        del primals_373
        # Topologically Sorted Source Nodes: [conv2d_186], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_374, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf832 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_186, x1_37], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf372, primals_375, buf832, 524288, grid=grid(524288), stream=stream0)
        buf373 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_148], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf371, buf372, primals_375, buf373, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_187], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf831 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_187, x2_37], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf374, primals_377, buf831, 524288, grid=grid(524288), stream=stream0)
        buf375 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_149], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf371, buf372, primals_375, buf374, primals_377, buf375, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_188], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_378, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf830 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_188, x3_37], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf376, primals_379, buf830, 524288, grid=grid(524288), stream=stream0)
        buf377 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_150], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf371, buf372, primals_375, buf374, primals_377, buf376, primals_379, buf377, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_189], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_380, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf829 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_189, x4_37], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf378, primals_381, buf829, 524288, grid=grid(524288), stream=stream0)
        buf379 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_151], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf371, buf372, primals_375, buf374, primals_377, buf376, primals_379, buf378, primals_381, buf379, 3145728, grid=grid(3145728), stream=stream0)
        del buf372
        del buf374
        del buf376
        del buf378
        del primals_375
        del primals_377
        del primals_379
        del primals_381
        # Topologically Sorted Source Nodes: [x5_37], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf381 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [x5_37, mul_49, out_37], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf381, primals_383, buf371, 1048576, grid=grid(1048576), stream=stream0)
        del primals_383
        # Topologically Sorted Source Nodes: [conv2d_191], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_384, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf828 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_191, x1_38], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf382, primals_385, buf828, 524288, grid=grid(524288), stream=stream0)
        buf383 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_152], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf381, buf382, primals_385, buf383, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_192], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf383, primals_386, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf827 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_192, x2_38], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf384, primals_387, buf827, 524288, grid=grid(524288), stream=stream0)
        buf385 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_153], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf381, buf382, primals_385, buf384, primals_387, buf385, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_193], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_388, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf826 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_193, x3_38], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf386, primals_389, buf826, 524288, grid=grid(524288), stream=stream0)
        buf387 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_154], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf381, buf382, primals_385, buf384, primals_387, buf386, primals_389, buf387, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_194], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf387, primals_390, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf825 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_194, x4_38], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf388, primals_391, buf825, 524288, grid=grid(524288), stream=stream0)
        buf389 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_155], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf381, buf382, primals_385, buf384, primals_387, buf386, primals_389, buf388, primals_391, buf389, 3145728, grid=grid(3145728), stream=stream0)
        del buf382
        del buf384
        del buf386
        del buf388
        del primals_385
        del primals_387
        del primals_389
        del primals_391
        # Topologically Sorted Source Nodes: [x5_38], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf391 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [x5_38, mul_50, out_38, mul_51, input_13], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf391, primals_393, buf381, buf361, 1048576, grid=grid(1048576), stream=stream0)
        del primals_393
        # Topologically Sorted Source Nodes: [conv2d_196], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_394, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf824 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_196, x1_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf392, primals_395, buf824, 524288, grid=grid(524288), stream=stream0)
        buf393 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_156], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf391, buf392, primals_395, buf393, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_197], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_396, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf823 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_197, x2_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf394, primals_397, buf823, 524288, grid=grid(524288), stream=stream0)
        buf395 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_157], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf391, buf392, primals_395, buf394, primals_397, buf395, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_198], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_398, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf822 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_198, x3_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf396, primals_399, buf822, 524288, grid=grid(524288), stream=stream0)
        buf397 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_158], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf391, buf392, primals_395, buf394, primals_397, buf396, primals_399, buf397, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_199], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_400, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf821 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_199, x4_39], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf398, primals_401, buf821, 524288, grid=grid(524288), stream=stream0)
        buf399 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_159], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf391, buf392, primals_395, buf394, primals_397, buf396, primals_399, buf398, primals_401, buf399, 3145728, grid=grid(3145728), stream=stream0)
        del buf392
        del buf394
        del buf396
        del buf398
        del primals_395
        del primals_397
        del primals_399
        del primals_401
        # Topologically Sorted Source Nodes: [x5_39], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf401 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [x5_39, mul_52, out_39], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf401, primals_403, buf391, 1048576, grid=grid(1048576), stream=stream0)
        del primals_403
        # Topologically Sorted Source Nodes: [conv2d_201], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_404, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf820 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_201, x1_40], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf402, primals_405, buf820, 524288, grid=grid(524288), stream=stream0)
        buf403 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_160], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf401, buf402, primals_405, buf403, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_202], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_406, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf819 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_202, x2_40], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf404, primals_407, buf819, 524288, grid=grid(524288), stream=stream0)
        buf405 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_161], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf401, buf402, primals_405, buf404, primals_407, buf405, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_203], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_408, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf818 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_203, x3_40], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf406, primals_409, buf818, 524288, grid=grid(524288), stream=stream0)
        buf407 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_162], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf401, buf402, primals_405, buf404, primals_407, buf406, primals_409, buf407, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_204], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, primals_410, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf817 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_204, x4_40], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf408, primals_411, buf817, 524288, grid=grid(524288), stream=stream0)
        buf409 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_163], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf401, buf402, primals_405, buf404, primals_407, buf406, primals_409, buf408, primals_411, buf409, 3145728, grid=grid(3145728), stream=stream0)
        del buf402
        del buf404
        del buf406
        del buf408
        del primals_405
        del primals_407
        del primals_409
        del primals_411
        # Topologically Sorted Source Nodes: [x5_40], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf411 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [x5_40, mul_53, out_40], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf411, primals_413, buf401, 1048576, grid=grid(1048576), stream=stream0)
        del primals_413
        # Topologically Sorted Source Nodes: [conv2d_206], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_414, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf816 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_206, x1_41], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf412, primals_415, buf816, 524288, grid=grid(524288), stream=stream0)
        buf413 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_164], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf411, buf412, primals_415, buf413, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_207], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_416, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf815 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_207, x2_41], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf414, primals_417, buf815, 524288, grid=grid(524288), stream=stream0)
        buf415 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_165], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf411, buf412, primals_415, buf414, primals_417, buf415, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_208], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_418, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf814 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_208, x3_41], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf416, primals_419, buf814, 524288, grid=grid(524288), stream=stream0)
        buf417 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_166], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf411, buf412, primals_415, buf414, primals_417, buf416, primals_419, buf417, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_209], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, primals_420, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf813 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_209, x4_41], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf418, primals_421, buf813, 524288, grid=grid(524288), stream=stream0)
        buf419 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_167], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf411, buf412, primals_415, buf414, primals_417, buf416, primals_419, buf418, primals_421, buf419, 3145728, grid=grid(3145728), stream=stream0)
        del buf412
        del buf414
        del buf416
        del buf418
        del primals_415
        del primals_417
        del primals_419
        del primals_421
        # Topologically Sorted Source Nodes: [x5_41], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf419, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf421 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [x5_41, mul_54, out_41, mul_55, input_14], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf421, primals_423, buf411, buf391, 1048576, grid=grid(1048576), stream=stream0)
        del primals_423
        # Topologically Sorted Source Nodes: [conv2d_211], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_424, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf812 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_211, x1_42], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf422, primals_425, buf812, 524288, grid=grid(524288), stream=stream0)
        buf423 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_168], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf421, buf422, primals_425, buf423, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_212], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_426, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf811 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_212, x2_42], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf424, primals_427, buf811, 524288, grid=grid(524288), stream=stream0)
        buf425 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_169], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf421, buf422, primals_425, buf424, primals_427, buf425, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_213], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_428, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf810 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_213, x3_42], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf426, primals_429, buf810, 524288, grid=grid(524288), stream=stream0)
        buf427 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_170], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf421, buf422, primals_425, buf424, primals_427, buf426, primals_429, buf427, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_214], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_430, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf809 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_214, x4_42], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf428, primals_431, buf809, 524288, grid=grid(524288), stream=stream0)
        buf429 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_171], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf421, buf422, primals_425, buf424, primals_427, buf426, primals_429, buf428, primals_431, buf429, 3145728, grid=grid(3145728), stream=stream0)
        del buf422
        del buf424
        del buf426
        del buf428
        del primals_425
        del primals_427
        del primals_429
        del primals_431
        # Topologically Sorted Source Nodes: [x5_42], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf431 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [x5_42, mul_56, out_42], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf431, primals_433, buf421, 1048576, grid=grid(1048576), stream=stream0)
        del primals_433
        # Topologically Sorted Source Nodes: [conv2d_216], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_434, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf808 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_216, x1_43], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf432, primals_435, buf808, 524288, grid=grid(524288), stream=stream0)
        buf433 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_172], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf431, buf432, primals_435, buf433, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_217], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_436, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf807 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_217, x2_43], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf434, primals_437, buf807, 524288, grid=grid(524288), stream=stream0)
        buf435 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_173], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf431, buf432, primals_435, buf434, primals_437, buf435, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_218], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, primals_438, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf806 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_218, x3_43], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf436, primals_439, buf806, 524288, grid=grid(524288), stream=stream0)
        buf437 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_174], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf431, buf432, primals_435, buf434, primals_437, buf436, primals_439, buf437, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_219], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_440, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf805 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_219, x4_43], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf438, primals_441, buf805, 524288, grid=grid(524288), stream=stream0)
        buf439 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_175], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf431, buf432, primals_435, buf434, primals_437, buf436, primals_439, buf438, primals_441, buf439, 3145728, grid=grid(3145728), stream=stream0)
        del buf432
        del buf434
        del buf436
        del buf438
        del primals_435
        del primals_437
        del primals_439
        del primals_441
        # Topologically Sorted Source Nodes: [x5_43], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf441 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [x5_43, mul_57, out_43], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf441, primals_443, buf431, 1048576, grid=grid(1048576), stream=stream0)
        del primals_443
        # Topologically Sorted Source Nodes: [conv2d_221], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, primals_444, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf804 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_221, x1_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf442, primals_445, buf804, 524288, grid=grid(524288), stream=stream0)
        buf443 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_176], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf441, buf442, primals_445, buf443, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_222], Original ATen: [aten.convolution]
        buf444 = extern_kernels.convolution(buf443, primals_446, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf803 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_222, x2_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf444, primals_447, buf803, 524288, grid=grid(524288), stream=stream0)
        buf445 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_177], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf441, buf442, primals_445, buf444, primals_447, buf445, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_223], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_448, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf802 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_223, x3_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf446, primals_449, buf802, 524288, grid=grid(524288), stream=stream0)
        buf447 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_178], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf441, buf442, primals_445, buf444, primals_447, buf446, primals_449, buf447, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_224], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_450, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf801 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_224, x4_44], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf448, primals_451, buf801, 524288, grid=grid(524288), stream=stream0)
        buf449 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_179], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf441, buf442, primals_445, buf444, primals_447, buf446, primals_449, buf448, primals_451, buf449, 3145728, grid=grid(3145728), stream=stream0)
        del buf442
        del buf444
        del buf446
        del buf448
        del primals_445
        del primals_447
        del primals_449
        del primals_451
        # Topologically Sorted Source Nodes: [x5_44], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, primals_452, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf451 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [x5_44, mul_58, out_44, mul_59, input_15], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf451, primals_453, buf441, buf421, 1048576, grid=grid(1048576), stream=stream0)
        del primals_453
        # Topologically Sorted Source Nodes: [conv2d_226], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, primals_454, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf800 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_226, x1_45], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf452, primals_455, buf800, 524288, grid=grid(524288), stream=stream0)
        buf453 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_180], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf451, buf452, primals_455, buf453, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_227], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, primals_456, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf799 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_227, x2_45], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf454, primals_457, buf799, 524288, grid=grid(524288), stream=stream0)
        buf455 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_181], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf451, buf452, primals_455, buf454, primals_457, buf455, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_228], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_458, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf798 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_228, x3_45], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf456, primals_459, buf798, 524288, grid=grid(524288), stream=stream0)
        buf457 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_182], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf451, buf452, primals_455, buf454, primals_457, buf456, primals_459, buf457, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_229], Original ATen: [aten.convolution]
        buf458 = extern_kernels.convolution(buf457, primals_460, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf458, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf797 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_229, x4_45], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf458, primals_461, buf797, 524288, grid=grid(524288), stream=stream0)
        buf459 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_183], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf451, buf452, primals_455, buf454, primals_457, buf456, primals_459, buf458, primals_461, buf459, 3145728, grid=grid(3145728), stream=stream0)
        del buf452
        del buf454
        del buf456
        del buf458
        del primals_455
        del primals_457
        del primals_459
        del primals_461
        # Topologically Sorted Source Nodes: [x5_45], Original ATen: [aten.convolution]
        buf460 = extern_kernels.convolution(buf459, primals_462, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf461 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [x5_45, mul_60, out_45], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf461, primals_463, buf451, 1048576, grid=grid(1048576), stream=stream0)
        del primals_463
        # Topologically Sorted Source Nodes: [conv2d_231], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_464, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf796 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_231, x1_46], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf462, primals_465, buf796, 524288, grid=grid(524288), stream=stream0)
        buf463 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_184], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf461, buf462, primals_465, buf463, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_232], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_466, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf795 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_232, x2_46], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf464, primals_467, buf795, 524288, grid=grid(524288), stream=stream0)
        buf465 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_185], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf461, buf462, primals_465, buf464, primals_467, buf465, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_233], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, primals_468, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf794 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_233, x3_46], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf466, primals_469, buf794, 524288, grid=grid(524288), stream=stream0)
        buf467 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_186], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf461, buf462, primals_465, buf464, primals_467, buf466, primals_469, buf467, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_234], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(buf467, primals_470, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf793 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_234, x4_46], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf468, primals_471, buf793, 524288, grid=grid(524288), stream=stream0)
        buf469 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_187], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf461, buf462, primals_465, buf464, primals_467, buf466, primals_469, buf468, primals_471, buf469, 3145728, grid=grid(3145728), stream=stream0)
        del buf462
        del buf464
        del buf466
        del buf468
        del primals_465
        del primals_467
        del primals_469
        del primals_471
        # Topologically Sorted Source Nodes: [x5_46], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_472, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf471 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [x5_46, mul_61, out_46], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf471, primals_473, buf461, 1048576, grid=grid(1048576), stream=stream0)
        del primals_473
        # Topologically Sorted Source Nodes: [conv2d_236], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, primals_474, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf792 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_236, x1_47], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf472, primals_475, buf792, 524288, grid=grid(524288), stream=stream0)
        buf473 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_188], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf471, buf472, primals_475, buf473, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_237], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, primals_476, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf791 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_237, x2_47], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf474, primals_477, buf791, 524288, grid=grid(524288), stream=stream0)
        buf475 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_189], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf471, buf472, primals_475, buf474, primals_477, buf475, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_238], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_478, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf790 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_238, x3_47], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf476, primals_479, buf790, 524288, grid=grid(524288), stream=stream0)
        buf477 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_190], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf471, buf472, primals_475, buf474, primals_477, buf476, primals_479, buf477, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_239], Original ATen: [aten.convolution]
        buf478 = extern_kernels.convolution(buf477, primals_480, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf478, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf789 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_239, x4_47], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf478, primals_481, buf789, 524288, grid=grid(524288), stream=stream0)
        buf479 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_191], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf471, buf472, primals_475, buf474, primals_477, buf476, primals_479, buf478, primals_481, buf479, 3145728, grid=grid(3145728), stream=stream0)
        del buf472
        del buf474
        del buf476
        del buf478
        del primals_475
        del primals_477
        del primals_479
        del primals_481
        # Topologically Sorted Source Nodes: [x5_47], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, primals_482, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf481 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [x5_47, mul_62, out_47, mul_63, input_16], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf481, primals_483, buf471, buf451, 1048576, grid=grid(1048576), stream=stream0)
        del primals_483
        # Topologically Sorted Source Nodes: [conv2d_241], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, primals_484, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf788 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_241, x1_48], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf482, primals_485, buf788, 524288, grid=grid(524288), stream=stream0)
        buf483 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_192], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf481, buf482, primals_485, buf483, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_242], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_486, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf787 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_242, x2_48], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf484, primals_487, buf787, 524288, grid=grid(524288), stream=stream0)
        buf485 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_193], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf481, buf482, primals_485, buf484, primals_487, buf485, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_243], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_488, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf786 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_243, x3_48], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf486, primals_489, buf786, 524288, grid=grid(524288), stream=stream0)
        buf487 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_194], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf481, buf482, primals_485, buf484, primals_487, buf486, primals_489, buf487, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_244], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf487, primals_490, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf785 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_244, x4_48], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf488, primals_491, buf785, 524288, grid=grid(524288), stream=stream0)
        buf489 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_195], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf481, buf482, primals_485, buf484, primals_487, buf486, primals_489, buf488, primals_491, buf489, 3145728, grid=grid(3145728), stream=stream0)
        del buf482
        del buf484
        del buf486
        del buf488
        del primals_485
        del primals_487
        del primals_489
        del primals_491
        # Topologically Sorted Source Nodes: [x5_48], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, primals_492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf491 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [x5_48, mul_64, out_48], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf491, primals_493, buf481, 1048576, grid=grid(1048576), stream=stream0)
        del primals_493
        # Topologically Sorted Source Nodes: [conv2d_246], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf491, primals_494, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf784 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_246, x1_49], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf492, primals_495, buf784, 524288, grid=grid(524288), stream=stream0)
        buf493 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_196], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf491, buf492, primals_495, buf493, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_247], Original ATen: [aten.convolution]
        buf494 = extern_kernels.convolution(buf493, primals_496, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf494, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf783 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_247, x2_49], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf494, primals_497, buf783, 524288, grid=grid(524288), stream=stream0)
        buf495 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_197], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf491, buf492, primals_495, buf494, primals_497, buf495, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_248], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf495, primals_498, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf782 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_248, x3_49], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf496, primals_499, buf782, 524288, grid=grid(524288), stream=stream0)
        buf497 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_198], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf491, buf492, primals_495, buf494, primals_497, buf496, primals_499, buf497, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_249], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_500, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf781 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_249, x4_49], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf498, primals_501, buf781, 524288, grid=grid(524288), stream=stream0)
        buf499 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_199], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf491, buf492, primals_495, buf494, primals_497, buf496, primals_499, buf498, primals_501, buf499, 3145728, grid=grid(3145728), stream=stream0)
        del buf492
        del buf494
        del buf496
        del buf498
        del primals_495
        del primals_497
        del primals_499
        del primals_501
        # Topologically Sorted Source Nodes: [x5_49], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, primals_502, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf501 = buf500; del buf500  # reuse
        # Topologically Sorted Source Nodes: [x5_49, mul_65, out_49], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf501, primals_503, buf491, 1048576, grid=grid(1048576), stream=stream0)
        del primals_503
        # Topologically Sorted Source Nodes: [conv2d_251], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_504, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf780 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_251, x1_50], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf502, primals_505, buf780, 524288, grid=grid(524288), stream=stream0)
        buf503 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_200], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf501, buf502, primals_505, buf503, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_252], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, primals_506, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf779 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_252, x2_50], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf504, primals_507, buf779, 524288, grid=grid(524288), stream=stream0)
        buf505 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_201], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf501, buf502, primals_505, buf504, primals_507, buf505, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_253], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_508, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf778 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_253, x3_50], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf506, primals_509, buf778, 524288, grid=grid(524288), stream=stream0)
        buf507 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_202], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf501, buf502, primals_505, buf504, primals_507, buf506, primals_509, buf507, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_254], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, primals_510, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf777 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_254, x4_50], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf508, primals_511, buf777, 524288, grid=grid(524288), stream=stream0)
        buf509 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_203], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf501, buf502, primals_505, buf504, primals_507, buf506, primals_509, buf508, primals_511, buf509, 3145728, grid=grid(3145728), stream=stream0)
        del buf502
        del buf504
        del buf506
        del buf508
        del primals_505
        del primals_507
        del primals_509
        del primals_511
        # Topologically Sorted Source Nodes: [x5_50], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, primals_512, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf511 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [x5_50, mul_66, out_50, mul_67, input_17], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf511, primals_513, buf501, buf481, 1048576, grid=grid(1048576), stream=stream0)
        del primals_513
        # Topologically Sorted Source Nodes: [conv2d_256], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_514, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf776 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_256, x1_51], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf512, primals_515, buf776, 524288, grid=grid(524288), stream=stream0)
        buf513 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_204], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf511, buf512, primals_515, buf513, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_257], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf513, primals_516, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf775 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_257, x2_51], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf514, primals_517, buf775, 524288, grid=grid(524288), stream=stream0)
        buf515 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_205], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf511, buf512, primals_515, buf514, primals_517, buf515, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_258], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, primals_518, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf774 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_258, x3_51], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf516, primals_519, buf774, 524288, grid=grid(524288), stream=stream0)
        buf517 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_206], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf511, buf512, primals_515, buf514, primals_517, buf516, primals_519, buf517, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_259], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, primals_520, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf773 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_259, x4_51], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf518, primals_521, buf773, 524288, grid=grid(524288), stream=stream0)
        buf519 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_207], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf511, buf512, primals_515, buf514, primals_517, buf516, primals_519, buf518, primals_521, buf519, 3145728, grid=grid(3145728), stream=stream0)
        del buf512
        del buf514
        del buf516
        del buf518
        del primals_515
        del primals_517
        del primals_519
        del primals_521
        # Topologically Sorted Source Nodes: [x5_51], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, primals_522, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf521 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [x5_51, mul_68, out_51], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf521, primals_523, buf511, 1048576, grid=grid(1048576), stream=stream0)
        del primals_523
        # Topologically Sorted Source Nodes: [conv2d_261], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_524, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf772 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_261, x1_52], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf522, primals_525, buf772, 524288, grid=grid(524288), stream=stream0)
        buf523 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_208], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf521, buf522, primals_525, buf523, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_262], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, primals_526, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf771 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_262, x2_52], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf524, primals_527, buf771, 524288, grid=grid(524288), stream=stream0)
        buf525 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_209], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf521, buf522, primals_525, buf524, primals_527, buf525, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_263], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_528, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf770 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_263, x3_52], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf526, primals_529, buf770, 524288, grid=grid(524288), stream=stream0)
        buf527 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_210], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf521, buf522, primals_525, buf524, primals_527, buf526, primals_529, buf527, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_264], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, primals_530, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf769 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_264, x4_52], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf528, primals_531, buf769, 524288, grid=grid(524288), stream=stream0)
        buf529 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_211], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf521, buf522, primals_525, buf524, primals_527, buf526, primals_529, buf528, primals_531, buf529, 3145728, grid=grid(3145728), stream=stream0)
        del buf522
        del buf524
        del buf526
        del buf528
        del primals_525
        del primals_527
        del primals_529
        del primals_531
        # Topologically Sorted Source Nodes: [x5_52], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf529, primals_532, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf531 = buf530; del buf530  # reuse
        # Topologically Sorted Source Nodes: [x5_52, mul_69, out_52], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf531, primals_533, buf521, 1048576, grid=grid(1048576), stream=stream0)
        del primals_533
        # Topologically Sorted Source Nodes: [conv2d_266], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf531, primals_534, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf768 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_266, x1_53], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf532, primals_535, buf768, 524288, grid=grid(524288), stream=stream0)
        buf533 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_212], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf531, buf532, primals_535, buf533, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_267], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_536, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf767 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_267, x2_53], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf534, primals_537, buf767, 524288, grid=grid(524288), stream=stream0)
        buf535 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_213], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf531, buf532, primals_535, buf534, primals_537, buf535, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_268], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_538, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf766 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_268, x3_53], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf536, primals_539, buf766, 524288, grid=grid(524288), stream=stream0)
        buf537 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_214], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf531, buf532, primals_535, buf534, primals_537, buf536, primals_539, buf537, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_269], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, primals_540, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf765 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_269, x4_53], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf538, primals_541, buf765, 524288, grid=grid(524288), stream=stream0)
        buf539 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_215], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf531, buf532, primals_535, buf534, primals_537, buf536, primals_539, buf538, primals_541, buf539, 3145728, grid=grid(3145728), stream=stream0)
        del buf532
        del buf534
        del buf536
        del buf538
        del primals_535
        del primals_537
        del primals_539
        del primals_541
        # Topologically Sorted Source Nodes: [x5_53], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, primals_542, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf541 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [x5_53, mul_70, out_53, mul_71, input_18], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf541, primals_543, buf531, buf511, 1048576, grid=grid(1048576), stream=stream0)
        del primals_543
        # Topologically Sorted Source Nodes: [conv2d_271], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_544, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf764 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_271, x1_54], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf542, primals_545, buf764, 524288, grid=grid(524288), stream=stream0)
        buf543 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_216], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf541, buf542, primals_545, buf543, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_272], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_546, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf544, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf763 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_272, x2_54], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf544, primals_547, buf763, 524288, grid=grid(524288), stream=stream0)
        buf545 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_217], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf541, buf542, primals_545, buf544, primals_547, buf545, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_273], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, primals_548, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf762 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_273, x3_54], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf546, primals_549, buf762, 524288, grid=grid(524288), stream=stream0)
        buf547 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_218], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf541, buf542, primals_545, buf544, primals_547, buf546, primals_549, buf547, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_274], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, primals_550, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf761 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_274, x4_54], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf548, primals_551, buf761, 524288, grid=grid(524288), stream=stream0)
        buf549 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_219], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf541, buf542, primals_545, buf544, primals_547, buf546, primals_549, buf548, primals_551, buf549, 3145728, grid=grid(3145728), stream=stream0)
        del buf542
        del buf544
        del buf546
        del buf548
        del primals_545
        del primals_547
        del primals_549
        del primals_551
        # Topologically Sorted Source Nodes: [x5_54], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, primals_552, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf551 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [x5_54, mul_72, out_54], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf551, primals_553, buf541, 1048576, grid=grid(1048576), stream=stream0)
        del primals_553
        # Topologically Sorted Source Nodes: [conv2d_276], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_554, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf760 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_276, x1_55], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf552, primals_555, buf760, 524288, grid=grid(524288), stream=stream0)
        buf553 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_220], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf551, buf552, primals_555, buf553, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_277], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, primals_556, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf759 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_277, x2_55], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf554, primals_557, buf759, 524288, grid=grid(524288), stream=stream0)
        buf555 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_221], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf551, buf552, primals_555, buf554, primals_557, buf555, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_278], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(buf555, primals_558, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf758 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_278, x3_55], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf556, primals_559, buf758, 524288, grid=grid(524288), stream=stream0)
        buf557 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_222], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf551, buf552, primals_555, buf554, primals_557, buf556, primals_559, buf557, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_279], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, primals_560, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf757 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_279, x4_55], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf558, primals_561, buf757, 524288, grid=grid(524288), stream=stream0)
        buf559 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_223], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf551, buf552, primals_555, buf554, primals_557, buf556, primals_559, buf558, primals_561, buf559, 3145728, grid=grid(3145728), stream=stream0)
        del buf552
        del buf554
        del buf556
        del buf558
        del primals_555
        del primals_557
        del primals_559
        del primals_561
        # Topologically Sorted Source Nodes: [x5_55], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_562, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf561 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [x5_55, mul_73, out_55], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf561, primals_563, buf551, 1048576, grid=grid(1048576), stream=stream0)
        del primals_563
        # Topologically Sorted Source Nodes: [conv2d_281], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf561, primals_564, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf756 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_281, x1_56], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf562, primals_565, buf756, 524288, grid=grid(524288), stream=stream0)
        buf563 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_224], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf561, buf562, primals_565, buf563, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_282], Original ATen: [aten.convolution]
        buf564 = extern_kernels.convolution(buf563, primals_566, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf564, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf755 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_282, x2_56], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf564, primals_567, buf755, 524288, grid=grid(524288), stream=stream0)
        buf565 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_225], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf561, buf562, primals_565, buf564, primals_567, buf565, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_283], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_568, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf754 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_283, x3_56], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf566, primals_569, buf754, 524288, grid=grid(524288), stream=stream0)
        buf567 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_226], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf561, buf562, primals_565, buf564, primals_567, buf566, primals_569, buf567, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_284], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf567, primals_570, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf568, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf753 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_284, x4_56], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf568, primals_571, buf753, 524288, grid=grid(524288), stream=stream0)
        buf569 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_227], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf561, buf562, primals_565, buf564, primals_567, buf566, primals_569, buf568, primals_571, buf569, 3145728, grid=grid(3145728), stream=stream0)
        del buf562
        del buf564
        del buf566
        del buf568
        del primals_565
        del primals_567
        del primals_569
        del primals_571
        # Topologically Sorted Source Nodes: [x5_56], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_572, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf571 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [x5_56, mul_74, out_56, mul_75, input_19], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf571, primals_573, buf561, buf541, 1048576, grid=grid(1048576), stream=stream0)
        del primals_573
        # Topologically Sorted Source Nodes: [conv2d_286], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_574, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf752 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_286, x1_57], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf572, primals_575, buf752, 524288, grid=grid(524288), stream=stream0)
        buf573 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_228], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf571, buf572, primals_575, buf573, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_287], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_576, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf751 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_287, x2_57], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf574, primals_577, buf751, 524288, grid=grid(524288), stream=stream0)
        buf575 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_229], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf571, buf572, primals_575, buf574, primals_577, buf575, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_288], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_578, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf750 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_288, x3_57], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf576, primals_579, buf750, 524288, grid=grid(524288), stream=stream0)
        buf577 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_230], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf571, buf572, primals_575, buf574, primals_577, buf576, primals_579, buf577, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_289], Original ATen: [aten.convolution]
        buf578 = extern_kernels.convolution(buf577, primals_580, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf749 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_289, x4_57], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf578, primals_581, buf749, 524288, grid=grid(524288), stream=stream0)
        buf579 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_231], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf571, buf572, primals_575, buf574, primals_577, buf576, primals_579, buf578, primals_581, buf579, 3145728, grid=grid(3145728), stream=stream0)
        del buf572
        del buf574
        del buf576
        del buf578
        del primals_575
        del primals_577
        del primals_579
        del primals_581
        # Topologically Sorted Source Nodes: [x5_57], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf579, primals_582, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf581 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [x5_57, mul_76, out_57], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf581, primals_583, buf571, 1048576, grid=grid(1048576), stream=stream0)
        del primals_583
        # Topologically Sorted Source Nodes: [conv2d_291], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, primals_584, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf748 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_291, x1_58], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf582, primals_585, buf748, 524288, grid=grid(524288), stream=stream0)
        buf583 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_232], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf581, buf582, primals_585, buf583, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_292], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf583, primals_586, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf747 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_292, x2_58], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf584, primals_587, buf747, 524288, grid=grid(524288), stream=stream0)
        buf585 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_233], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf581, buf582, primals_585, buf584, primals_587, buf585, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_293], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_588, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf746 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_293, x3_58], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf586, primals_589, buf746, 524288, grid=grid(524288), stream=stream0)
        buf587 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_234], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf581, buf582, primals_585, buf584, primals_587, buf586, primals_589, buf587, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_294], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf587, primals_590, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf745 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_294, x4_58], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf588, primals_591, buf745, 524288, grid=grid(524288), stream=stream0)
        buf589 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_235], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf581, buf582, primals_585, buf584, primals_587, buf586, primals_589, buf588, primals_591, buf589, 3145728, grid=grid(3145728), stream=stream0)
        del buf582
        del buf584
        del buf586
        del buf588
        del primals_585
        del primals_587
        del primals_589
        del primals_591
        # Topologically Sorted Source Nodes: [x5_58], Original ATen: [aten.convolution]
        buf590 = extern_kernels.convolution(buf589, primals_592, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf590, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf591 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [x5_58, mul_77, out_58], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf591, primals_593, buf581, 1048576, grid=grid(1048576), stream=stream0)
        del primals_593
        # Topologically Sorted Source Nodes: [conv2d_296], Original ATen: [aten.convolution]
        buf592 = extern_kernels.convolution(buf591, primals_594, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf744 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_296, x1_59], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf592, primals_595, buf744, 524288, grid=grid(524288), stream=stream0)
        buf593 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_236], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf591, buf592, primals_595, buf593, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_297], Original ATen: [aten.convolution]
        buf594 = extern_kernels.convolution(buf593, primals_596, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf594, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf743 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_297, x2_59], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf594, primals_597, buf743, 524288, grid=grid(524288), stream=stream0)
        buf595 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_237], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf591, buf592, primals_595, buf594, primals_597, buf595, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_298], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_598, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf742 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_298, x3_59], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf596, primals_599, buf742, 524288, grid=grid(524288), stream=stream0)
        buf597 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_238], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf591, buf592, primals_595, buf594, primals_597, buf596, primals_599, buf597, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_299], Original ATen: [aten.convolution]
        buf598 = extern_kernels.convolution(buf597, primals_600, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf741 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_299, x4_59], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf598, primals_601, buf741, 524288, grid=grid(524288), stream=stream0)
        buf599 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_239], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf591, buf592, primals_595, buf594, primals_597, buf596, primals_599, buf598, primals_601, buf599, 3145728, grid=grid(3145728), stream=stream0)
        del buf592
        del buf594
        del buf596
        del buf598
        del primals_595
        del primals_597
        del primals_599
        del primals_601
        # Topologically Sorted Source Nodes: [x5_59], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, primals_602, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf601 = buf600; del buf600  # reuse
        # Topologically Sorted Source Nodes: [x5_59, mul_78, out_59, mul_79, input_20], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf601, primals_603, buf591, buf571, 1048576, grid=grid(1048576), stream=stream0)
        del primals_603
        # Topologically Sorted Source Nodes: [conv2d_301], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, primals_604, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf740 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_301, x1_60], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf602, primals_605, buf740, 524288, grid=grid(524288), stream=stream0)
        buf603 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_240], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf601, buf602, primals_605, buf603, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_302], Original ATen: [aten.convolution]
        buf604 = extern_kernels.convolution(buf603, primals_606, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf604, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf739 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_302, x2_60], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf604, primals_607, buf739, 524288, grid=grid(524288), stream=stream0)
        buf605 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_241], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf601, buf602, primals_605, buf604, primals_607, buf605, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_303], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_608, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf738 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_303, x3_60], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf606, primals_609, buf738, 524288, grid=grid(524288), stream=stream0)
        buf607 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_242], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf601, buf602, primals_605, buf604, primals_607, buf606, primals_609, buf607, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_304], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(buf607, primals_610, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf737 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_304, x4_60], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf608, primals_611, buf737, 524288, grid=grid(524288), stream=stream0)
        buf609 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_243], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf601, buf602, primals_605, buf604, primals_607, buf606, primals_609, buf608, primals_611, buf609, 3145728, grid=grid(3145728), stream=stream0)
        del buf602
        del buf604
        del buf606
        del buf608
        del primals_605
        del primals_607
        del primals_609
        del primals_611
        # Topologically Sorted Source Nodes: [x5_60], Original ATen: [aten.convolution]
        buf610 = extern_kernels.convolution(buf609, primals_612, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf610, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf611 = buf610; del buf610  # reuse
        # Topologically Sorted Source Nodes: [x5_60, mul_80, out_60], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf611, primals_613, buf601, 1048576, grid=grid(1048576), stream=stream0)
        del primals_613
        # Topologically Sorted Source Nodes: [conv2d_306], Original ATen: [aten.convolution]
        buf612 = extern_kernels.convolution(buf611, primals_614, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf612, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf736 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_306, x1_61], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf612, primals_615, buf736, 524288, grid=grid(524288), stream=stream0)
        buf613 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_244], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf611, buf612, primals_615, buf613, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_307], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf613, primals_616, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf735 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_307, x2_61], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf614, primals_617, buf735, 524288, grid=grid(524288), stream=stream0)
        buf615 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_245], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf611, buf612, primals_615, buf614, primals_617, buf615, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_308], Original ATen: [aten.convolution]
        buf616 = extern_kernels.convolution(buf615, primals_618, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf616, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf734 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_308, x3_61], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf616, primals_619, buf734, 524288, grid=grid(524288), stream=stream0)
        buf617 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_246], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf611, buf612, primals_615, buf614, primals_617, buf616, primals_619, buf617, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_309], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, primals_620, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf733 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_309, x4_61], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf618, primals_621, buf733, 524288, grid=grid(524288), stream=stream0)
        buf619 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_247], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf611, buf612, primals_615, buf614, primals_617, buf616, primals_619, buf618, primals_621, buf619, 3145728, grid=grid(3145728), stream=stream0)
        del buf612
        del buf614
        del buf616
        del buf618
        del primals_615
        del primals_617
        del primals_619
        del primals_621
        # Topologically Sorted Source Nodes: [x5_61], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_622, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf621 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [x5_61, mul_81, out_61], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf621, primals_623, buf611, 1048576, grid=grid(1048576), stream=stream0)
        del primals_623
        # Topologically Sorted Source Nodes: [conv2d_311], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf621, primals_624, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf622, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf732 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_311, x1_62], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf622, primals_625, buf732, 524288, grid=grid(524288), stream=stream0)
        buf623 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_248], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf621, buf622, primals_625, buf623, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_312], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf623, primals_626, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf731 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_312, x2_62], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf624, primals_627, buf731, 524288, grid=grid(524288), stream=stream0)
        buf625 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_249], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf621, buf622, primals_625, buf624, primals_627, buf625, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_313], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf625, primals_628, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf626, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf730 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_313, x3_62], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf626, primals_629, buf730, 524288, grid=grid(524288), stream=stream0)
        buf627 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_250], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf621, buf622, primals_625, buf624, primals_627, buf626, primals_629, buf627, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_314], Original ATen: [aten.convolution]
        buf628 = extern_kernels.convolution(buf627, primals_630, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf628, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf729 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_314, x4_62], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf628, primals_631, buf729, 524288, grid=grid(524288), stream=stream0)
        buf629 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_251], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf621, buf622, primals_625, buf624, primals_627, buf626, primals_629, buf628, primals_631, buf629, 3145728, grid=grid(3145728), stream=stream0)
        del buf622
        del buf624
        del buf626
        del buf628
        del primals_625
        del primals_627
        del primals_629
        del primals_631
        # Topologically Sorted Source Nodes: [x5_62], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, primals_632, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf631 = buf630; del buf630  # reuse
        # Topologically Sorted Source Nodes: [x5_62, mul_82, out_62, mul_83, input_21], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf631, primals_633, buf621, buf601, 1048576, grid=grid(1048576), stream=stream0)
        del primals_633
        # Topologically Sorted Source Nodes: [conv2d_316], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_634, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf728 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_316, x1_63], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf632, primals_635, buf728, 524288, grid=grid(524288), stream=stream0)
        buf633 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_252], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf631, buf632, primals_635, buf633, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_317], Original ATen: [aten.convolution]
        buf634 = extern_kernels.convolution(buf633, primals_636, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf634, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf727 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_317, x2_63], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf634, primals_637, buf727, 524288, grid=grid(524288), stream=stream0)
        buf635 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_253], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf631, buf632, primals_635, buf634, primals_637, buf635, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_318], Original ATen: [aten.convolution]
        buf636 = extern_kernels.convolution(buf635, primals_638, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf636, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf726 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_318, x3_63], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf636, primals_639, buf726, 524288, grid=grid(524288), stream=stream0)
        buf637 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_254], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf631, buf632, primals_635, buf634, primals_637, buf636, primals_639, buf637, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_319], Original ATen: [aten.convolution]
        buf638 = extern_kernels.convolution(buf637, primals_640, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf638, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf725 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_319, x4_63], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf638, primals_641, buf725, 524288, grid=grid(524288), stream=stream0)
        buf639 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_255], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf631, buf632, primals_635, buf634, primals_637, buf636, primals_639, buf638, primals_641, buf639, 3145728, grid=grid(3145728), stream=stream0)
        del buf632
        del buf634
        del buf636
        del buf638
        del primals_635
        del primals_637
        del primals_639
        del primals_641
        # Topologically Sorted Source Nodes: [x5_63], Original ATen: [aten.convolution]
        buf640 = extern_kernels.convolution(buf639, primals_642, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf640, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf641 = buf640; del buf640  # reuse
        # Topologically Sorted Source Nodes: [x5_63, mul_84, out_63], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf641, primals_643, buf631, 1048576, grid=grid(1048576), stream=stream0)
        del primals_643
        # Topologically Sorted Source Nodes: [conv2d_321], Original ATen: [aten.convolution]
        buf642 = extern_kernels.convolution(buf641, primals_644, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf724 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_321, x1_64], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf642, primals_645, buf724, 524288, grid=grid(524288), stream=stream0)
        buf643 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_256], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf641, buf642, primals_645, buf643, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_322], Original ATen: [aten.convolution]
        buf644 = extern_kernels.convolution(buf643, primals_646, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf644, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf723 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_322, x2_64], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf644, primals_647, buf723, 524288, grid=grid(524288), stream=stream0)
        buf645 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_257], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf641, buf642, primals_645, buf644, primals_647, buf645, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_323], Original ATen: [aten.convolution]
        buf646 = extern_kernels.convolution(buf645, primals_648, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf722 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_323, x3_64], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf646, primals_649, buf722, 524288, grid=grid(524288), stream=stream0)
        buf647 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_258], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf641, buf642, primals_645, buf644, primals_647, buf646, primals_649, buf647, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_324], Original ATen: [aten.convolution]
        buf648 = extern_kernels.convolution(buf647, primals_650, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf648, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf721 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_324, x4_64], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf648, primals_651, buf721, 524288, grid=grid(524288), stream=stream0)
        buf649 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_259], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf641, buf642, primals_645, buf644, primals_647, buf646, primals_649, buf648, primals_651, buf649, 3145728, grid=grid(3145728), stream=stream0)
        del buf642
        del buf644
        del buf646
        del buf648
        del primals_645
        del primals_647
        del primals_649
        del primals_651
        # Topologically Sorted Source Nodes: [x5_64], Original ATen: [aten.convolution]
        buf650 = extern_kernels.convolution(buf649, primals_652, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf650, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf651 = buf650; del buf650  # reuse
        # Topologically Sorted Source Nodes: [x5_64, mul_85, out_64], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf651, primals_653, buf641, 1048576, grid=grid(1048576), stream=stream0)
        del primals_653
        # Topologically Sorted Source Nodes: [conv2d_326], Original ATen: [aten.convolution]
        buf652 = extern_kernels.convolution(buf651, primals_654, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf652, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf720 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_326, x1_65], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf652, primals_655, buf720, 524288, grid=grid(524288), stream=stream0)
        buf653 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_260], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf651, buf652, primals_655, buf653, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_327], Original ATen: [aten.convolution]
        buf654 = extern_kernels.convolution(buf653, primals_656, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf719 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_327, x2_65], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf654, primals_657, buf719, 524288, grid=grid(524288), stream=stream0)
        buf655 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_261], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf651, buf652, primals_655, buf654, primals_657, buf655, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_328], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_658, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf718 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_328, x3_65], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf656, primals_659, buf718, 524288, grid=grid(524288), stream=stream0)
        buf657 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_262], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf651, buf652, primals_655, buf654, primals_657, buf656, primals_659, buf657, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_329], Original ATen: [aten.convolution]
        buf658 = extern_kernels.convolution(buf657, primals_660, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf658, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf717 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_329, x4_65], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf658, primals_661, buf717, 524288, grid=grid(524288), stream=stream0)
        buf659 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_263], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf651, buf652, primals_655, buf654, primals_657, buf656, primals_659, buf658, primals_661, buf659, 3145728, grid=grid(3145728), stream=stream0)
        del buf652
        del buf654
        del buf656
        del buf658
        del primals_655
        del primals_657
        del primals_659
        del primals_661
        # Topologically Sorted Source Nodes: [x5_65], Original ATen: [aten.convolution]
        buf660 = extern_kernels.convolution(buf659, primals_662, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf660, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf661 = buf660; del buf660  # reuse
        # Topologically Sorted Source Nodes: [x5_65, mul_86, out_65, mul_87, input_22], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf661, primals_663, buf651, buf631, 1048576, grid=grid(1048576), stream=stream0)
        del primals_663
        # Topologically Sorted Source Nodes: [conv2d_331], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, primals_664, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf716 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_331, x1_66], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf662, primals_665, buf716, 524288, grid=grid(524288), stream=stream0)
        buf663 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_264], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf661, buf662, primals_665, buf663, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_332], Original ATen: [aten.convolution]
        buf664 = extern_kernels.convolution(buf663, primals_666, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf664, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf715 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_332, x2_66], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf664, primals_667, buf715, 524288, grid=grid(524288), stream=stream0)
        buf665 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_265], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf661, buf662, primals_665, buf664, primals_667, buf665, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_333], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, primals_668, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf714 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_333, x3_66], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf666, primals_669, buf714, 524288, grid=grid(524288), stream=stream0)
        buf667 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_266], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf661, buf662, primals_665, buf664, primals_667, buf666, primals_669, buf667, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_334], Original ATen: [aten.convolution]
        buf668 = extern_kernels.convolution(buf667, primals_670, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf713 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_334, x4_66], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf668, primals_671, buf713, 524288, grid=grid(524288), stream=stream0)
        buf669 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_267], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf661, buf662, primals_665, buf664, primals_667, buf666, primals_669, buf668, primals_671, buf669, 3145728, grid=grid(3145728), stream=stream0)
        del buf662
        del buf664
        del buf666
        del buf668
        del primals_665
        del primals_667
        del primals_669
        del primals_671
        # Topologically Sorted Source Nodes: [x5_66], Original ATen: [aten.convolution]
        buf670 = extern_kernels.convolution(buf669, primals_672, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf670, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf671 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [x5_66, mul_88, out_66], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf671, primals_673, buf661, 1048576, grid=grid(1048576), stream=stream0)
        del primals_673
        # Topologically Sorted Source Nodes: [conv2d_336], Original ATen: [aten.convolution]
        buf672 = extern_kernels.convolution(buf671, primals_674, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf672, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf712 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_336, x1_67], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf672, primals_675, buf712, 524288, grid=grid(524288), stream=stream0)
        buf673 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_268], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf671, buf672, primals_675, buf673, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_337], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, primals_676, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf711 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_337, x2_67], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf674, primals_677, buf711, 524288, grid=grid(524288), stream=stream0)
        buf675 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_269], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf671, buf672, primals_675, buf674, primals_677, buf675, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_338], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, primals_678, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf710 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_338, x3_67], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf676, primals_679, buf710, 524288, grid=grid(524288), stream=stream0)
        buf677 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_270], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf671, buf672, primals_675, buf674, primals_677, buf676, primals_679, buf677, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_339], Original ATen: [aten.convolution]
        buf678 = extern_kernels.convolution(buf677, primals_680, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf678, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf709 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_339, x4_67], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf678, primals_681, buf709, 524288, grid=grid(524288), stream=stream0)
        buf679 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_271], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf671, buf672, primals_675, buf674, primals_677, buf676, primals_679, buf678, primals_681, buf679, 3145728, grid=grid(3145728), stream=stream0)
        del buf672
        del buf674
        del buf676
        del buf678
        del primals_675
        del primals_677
        del primals_679
        del primals_681
        # Topologically Sorted Source Nodes: [x5_67], Original ATen: [aten.convolution]
        buf680 = extern_kernels.convolution(buf679, primals_682, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf680, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf681 = buf680; del buf680  # reuse
        # Topologically Sorted Source Nodes: [x5_67, mul_89, out_67], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_8.run(buf681, primals_683, buf671, 1048576, grid=grid(1048576), stream=stream0)
        del primals_683
        # Topologically Sorted Source Nodes: [conv2d_341], Original ATen: [aten.convolution]
        buf682 = extern_kernels.convolution(buf681, primals_684, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf682, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf708 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_341, x1_68], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf682, primals_685, buf708, 524288, grid=grid(524288), stream=stream0)
        buf683 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_272], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf681, buf682, primals_685, buf683, 1572864, grid=grid(1572864), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_342], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, primals_686, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf684, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf707 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_342, x2_68], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf684, primals_687, buf707, 524288, grid=grid(524288), stream=stream0)
        buf685 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_273], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf681, buf682, primals_685, buf684, primals_687, buf685, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_343], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf685, primals_688, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf706 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_343, x3_68], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf686, primals_689, buf706, 524288, grid=grid(524288), stream=stream0)
        buf687 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_274], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf681, buf682, primals_685, buf684, primals_687, buf686, primals_689, buf687, 2621440, grid=grid(2621440), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_344], Original ATen: [aten.convolution]
        buf688 = extern_kernels.convolution(buf687, primals_690, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf688, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf705 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_344, x4_68], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_3.run(buf688, primals_691, buf705, 524288, grid=grid(524288), stream=stream0)
        buf689 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_275], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf681, buf682, primals_685, buf684, primals_687, buf686, primals_689, buf688, primals_691, buf689, 3145728, grid=grid(3145728), stream=stream0)
        del buf682
        del buf684
        del buf686
        del buf688
        del primals_685
        del primals_687
        del primals_689
        del primals_691
        # Topologically Sorted Source Nodes: [x5_68], Original ATen: [aten.convolution]
        buf690 = extern_kernels.convolution(buf689, primals_692, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf690, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf691 = buf690; del buf690  # reuse
        # Topologically Sorted Source Nodes: [x5_68, mul_90, out_68, mul_91, input_23], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_9.run(buf691, primals_693, buf681, buf661, 1048576, grid=grid(1048576), stream=stream0)
        del primals_693
        # Topologically Sorted Source Nodes: [trunk], Original ATen: [aten.convolution]
        buf692 = extern_kernels.convolution(buf691, primals_694, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf692, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf694 = empty_strided_cuda((4, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [trunk, fea_1, interpolate], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_10.run(buf693, buf1, buf692, primals_695, buf694, 4194304, grid=grid(4194304), stream=stream0)
        del buf692
        del primals_695
        # Topologically Sorted Source Nodes: [conv2d_347], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(buf694, primals_696, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf695, (4, 64, 128, 128), (1048576, 16384, 128, 1))
        buf704 = empty_strided_cuda((4, 64, 128, 128), (1048576, 16384, 128, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_347, fea_2], Original ATen: [aten.convolution, aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_leaky_relu_backward_11.run(buf695, primals_697, buf704, 4194304, grid=grid(4194304), stream=stream0)
        buf697 = empty_strided_cuda((4, 64, 256, 256), (4194304, 65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_347, fea_2, interpolate_1], Original ATen: [aten.convolution, aten.leaky_relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_convolution_leaky_relu_12.run(buf696, buf695, primals_697, buf697, 16777216, grid=grid(16777216), stream=stream0)
        del buf695
        del primals_697
        # Topologically Sorted Source Nodes: [conv2d_348], Original ATen: [aten.convolution]
        buf698 = extern_kernels.convolution(buf697, primals_698, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf698, (4, 64, 256, 256), (4194304, 65536, 256, 1))
        buf699 = buf698; del buf698  # reuse
        # Topologically Sorted Source Nodes: [conv2d_348, fea_3], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_13.run(buf699, primals_699, 16777216, grid=grid(16777216), stream=stream0)
        del primals_699
        # Topologically Sorted Source Nodes: [conv2d_349], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf699, primals_700, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf700, (4, 64, 256, 256), (4194304, 65536, 256, 1))
        buf701 = buf700; del buf700  # reuse
        # Topologically Sorted Source Nodes: [conv2d_349, leaky_relu_278], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_13.run(buf701, primals_701, 16777216, grid=grid(16777216), stream=stream0)
        del primals_701
        # Topologically Sorted Source Nodes: [out_69], Original ATen: [aten.convolution]
        buf702 = extern_kernels.convolution(buf701, primals_702, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf702, (4, 3, 256, 256), (196608, 65536, 256, 1))
        buf703 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [out_69], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(buf703, primals_703, 786432, grid=grid(786432), stream=stream0)
        del primals_703
    return (buf703, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, primals_232, primals_234, primals_236, primals_238, primals_240, primals_242, primals_244, primals_246, primals_248, primals_250, primals_252, primals_254, primals_256, primals_258, primals_260, primals_262, primals_264, primals_266, primals_268, primals_270, primals_272, primals_274, primals_276, primals_278, primals_280, primals_282, primals_284, primals_286, primals_288, primals_290, primals_292, primals_294, primals_296, primals_298, primals_300, primals_302, primals_304, primals_306, primals_308, primals_310, primals_312, primals_314, primals_316, primals_318, primals_320, primals_322, primals_324, primals_326, primals_328, primals_330, primals_332, primals_334, primals_336, primals_338, primals_340, primals_342, primals_344, primals_346, primals_348, primals_350, primals_352, primals_354, primals_356, primals_358, primals_360, primals_362, primals_364, primals_366, primals_368, primals_370, primals_372, primals_374, primals_376, primals_378, primals_380, primals_382, primals_384, primals_386, primals_388, primals_390, primals_392, primals_394, primals_396, primals_398, primals_400, primals_402, primals_404, primals_406, primals_408, primals_410, primals_412, primals_414, primals_416, primals_418, primals_420, primals_422, primals_424, primals_426, primals_428, primals_430, primals_432, primals_434, primals_436, primals_438, primals_440, primals_442, primals_444, primals_446, primals_448, primals_450, primals_452, primals_454, primals_456, primals_458, primals_460, primals_462, primals_464, primals_466, primals_468, primals_470, primals_472, primals_474, primals_476, primals_478, primals_480, primals_482, primals_484, primals_486, primals_488, primals_490, primals_492, primals_494, primals_496, primals_498, primals_500, primals_502, primals_504, primals_506, primals_508, primals_510, primals_512, primals_514, primals_516, primals_518, primals_520, primals_522, primals_524, primals_526, primals_528, primals_530, primals_532, primals_534, primals_536, primals_538, primals_540, primals_542, primals_544, primals_546, primals_548, primals_550, primals_552, primals_554, primals_556, primals_558, primals_560, primals_562, primals_564, primals_566, primals_568, primals_570, primals_572, primals_574, primals_576, primals_578, primals_580, primals_582, primals_584, primals_586, primals_588, primals_590, primals_592, primals_594, primals_596, primals_598, primals_600, primals_602, primals_604, primals_606, primals_608, primals_610, primals_612, primals_614, primals_616, primals_618, primals_620, primals_622, primals_624, primals_626, primals_628, primals_630, primals_632, primals_634, primals_636, primals_638, primals_640, primals_642, primals_644, primals_646, primals_648, primals_650, primals_652, primals_654, primals_656, primals_658, primals_660, primals_662, primals_664, primals_666, primals_668, primals_670, primals_672, primals_674, primals_676, primals_678, primals_680, primals_682, primals_684, primals_686, primals_688, primals_690, primals_692, primals_694, primals_696, primals_698, primals_700, primals_702, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf69, buf71, buf73, buf75, buf77, buf79, buf81, buf83, buf85, buf87, buf89, buf91, buf93, buf95, buf97, buf99, buf101, buf103, buf105, buf107, buf109, buf111, buf113, buf115, buf117, buf119, buf121, buf123, buf125, buf127, buf129, buf131, buf133, buf135, buf137, buf139, buf141, buf143, buf145, buf147, buf149, buf151, buf153, buf155, buf157, buf159, buf161, buf163, buf165, buf167, buf169, buf171, buf173, buf175, buf177, buf179, buf181, buf183, buf185, buf187, buf189, buf191, buf193, buf195, buf197, buf199, buf201, buf203, buf205, buf207, buf209, buf211, buf213, buf215, buf217, buf219, buf221, buf223, buf225, buf227, buf229, buf231, buf233, buf235, buf237, buf239, buf241, buf243, buf245, buf247, buf249, buf251, buf253, buf255, buf257, buf259, buf261, buf263, buf265, buf267, buf269, buf271, buf273, buf275, buf277, buf279, buf281, buf283, buf285, buf287, buf289, buf291, buf293, buf295, buf297, buf299, buf301, buf303, buf305, buf307, buf309, buf311, buf313, buf315, buf317, buf319, buf321, buf323, buf325, buf327, buf329, buf331, buf333, buf335, buf337, buf339, buf341, buf343, buf345, buf347, buf349, buf351, buf353, buf355, buf357, buf359, buf361, buf363, buf365, buf367, buf369, buf371, buf373, buf375, buf377, buf379, buf381, buf383, buf385, buf387, buf389, buf391, buf393, buf395, buf397, buf399, buf401, buf403, buf405, buf407, buf409, buf411, buf413, buf415, buf417, buf419, buf421, buf423, buf425, buf427, buf429, buf431, buf433, buf435, buf437, buf439, buf441, buf443, buf445, buf447, buf449, buf451, buf453, buf455, buf457, buf459, buf461, buf463, buf465, buf467, buf469, buf471, buf473, buf475, buf477, buf479, buf481, buf483, buf485, buf487, buf489, buf491, buf493, buf495, buf497, buf499, buf501, buf503, buf505, buf507, buf509, buf511, buf513, buf515, buf517, buf519, buf521, buf523, buf525, buf527, buf529, buf531, buf533, buf535, buf537, buf539, buf541, buf543, buf545, buf547, buf549, buf551, buf553, buf555, buf557, buf559, buf561, buf563, buf565, buf567, buf569, buf571, buf573, buf575, buf577, buf579, buf581, buf583, buf585, buf587, buf589, buf591, buf593, buf595, buf597, buf599, buf601, buf603, buf605, buf607, buf609, buf611, buf613, buf615, buf617, buf619, buf621, buf623, buf625, buf627, buf629, buf631, buf633, buf635, buf637, buf639, buf641, buf643, buf645, buf647, buf649, buf651, buf653, buf655, buf657, buf659, buf661, buf663, buf665, buf667, buf669, buf671, buf673, buf675, buf677, buf679, buf681, buf683, buf685, buf687, buf689, buf691, buf693, buf694, buf696, buf697, buf699, buf701, buf704, buf705, buf706, buf707, buf708, buf709, buf710, buf711, buf712, buf713, buf714, buf715, buf716, buf717, buf718, buf719, buf720, buf721, buf722, buf723, buf724, buf725, buf726, buf727, buf728, buf729, buf730, buf731, buf732, buf733, buf734, buf735, buf736, buf737, buf738, buf739, buf740, buf741, buf742, buf743, buf744, buf745, buf746, buf747, buf748, buf749, buf750, buf751, buf752, buf753, buf754, buf755, buf756, buf757, buf758, buf759, buf760, buf761, buf762, buf763, buf764, buf765, buf766, buf767, buf768, buf769, buf770, buf771, buf772, buf773, buf774, buf775, buf776, buf777, buf778, buf779, buf780, buf781, buf782, buf783, buf784, buf785, buf786, buf787, buf788, buf789, buf790, buf791, buf792, buf793, buf794, buf795, buf796, buf797, buf798, buf799, buf800, buf801, buf802, buf803, buf804, buf805, buf806, buf807, buf808, buf809, buf810, buf811, buf812, buf813, buf814, buf815, buf816, buf817, buf818, buf819, buf820, buf821, buf822, buf823, buf824, buf825, buf826, buf827, buf828, buf829, buf830, buf831, buf832, buf833, buf834, buf835, buf836, buf837, buf838, buf839, buf840, buf841, buf842, buf843, buf844, buf845, buf846, buf847, buf848, buf849, buf850, buf851, buf852, buf853, buf854, buf855, buf856, buf857, buf858, buf859, buf860, buf861, buf862, buf863, buf864, buf865, buf866, buf867, buf868, buf869, buf870, buf871, buf872, buf873, buf874, buf875, buf876, buf877, buf878, buf879, buf880, buf881, buf882, buf883, buf884, buf885, buf886, buf887, buf888, buf889, buf890, buf891, buf892, buf893, buf894, buf895, buf896, buf897, buf898, buf899, buf900, buf901, buf902, buf903, buf904, buf905, buf906, buf907, buf908, buf909, buf910, buf911, buf912, buf913, buf914, buf915, buf916, buf917, buf918, buf919, buf920, buf921, buf922, buf923, buf924, buf925, buf926, buf927, buf928, buf929, buf930, buf931, buf932, buf933, buf934, buf935, buf936, buf937, buf938, buf939, buf940, buf941, buf942, buf943, buf944, buf945, buf946, buf947, buf948, buf949, buf950, buf951, buf952, buf953, buf954, buf955, buf956, buf957, buf958, buf959, buf960, buf961, buf962, buf963, buf964, buf965, buf966, buf967, buf968, buf969, buf970, buf971, buf972, buf973, buf974, buf975, buf976, buf977, buf978, buf979, buf980, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((3, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
