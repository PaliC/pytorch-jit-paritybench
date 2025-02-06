# AOT ID: ['23_forward']
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


# kernel path: inductor_cache/iw/ciwxndmmc5v322h6nokodrnyjqovt44rsfhvd4kffrbtdyyjotoj.py
# Topologically Sorted Source Nodes: [var_mean, add, sqrt], Original ATen: [aten.var_mean, aten.add, aten.sqrt]
# Source node to ATen node mapping:
#   add => add
#   sqrt => sqrt
#   var_mean => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [1]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
triton_poi_fused_add_sqrt_var_mean_0 = async_compile.triton('triton_poi_fused_add_sqrt_var_mean_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sqrt_var_mean_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sqrt_var_mean_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.sqrt(tmp22)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fg/cfgytv5otxabirh5bvpgadjf7uuxifgs3tzrdgzjsp2aeluylymn.py
# Topologically Sorted Source Nodes: [var_mean, sub, add, sqrt, x, mul, add_1, y], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   mul => mul
#   sqrt => sqrt
#   sub => sub
#   var_mean => var_mean
#   x => div
#   y => relu
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %sqrt), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %primals_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %primals_3), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_1 = async_compile.triton('triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zg/czg24sttpgahblg4s7jbuxgjsrvhgqtfxqmc6ypambqvf7xaignl.py
# Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   y_1 => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_4, %primals_5, [1], [0], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oq/coqh5bdglpsd6coqkunp7zmvbzpp5xgknljkrc3e2ziplbtwklha.py
# Topologically Sorted Source Nodes: [var_mean_1], Original ATen: [aten.var_mean]
# Source node to ATen node mapping:
#   var_mean_1 => var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [1]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_var_mean_3 = async_compile.triton('triton_poi_fused_var_mean_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_var_mean_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_var_mean_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n5/cn5vf4biahpydyjvq6y27zxttd6j5xgdkpe2uufiu5m334erbiqb.py
# Topologically Sorted Source Nodes: [var_mean_1, sub_1, add_2, sqrt_1, x_1, mul_1, add_3, y_2, x_2], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   mul_1 => mul_1
#   sqrt_1 => sqrt_1
#   sub_1 => sub_1
#   var_mean_1 => var_mean_1
#   x_1 => div_1
#   x_2 => constant_pad_nd
#   y_2 => relu_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %sqrt_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %primals_6), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_7), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_1, [3, 0], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_div_mul_relu_sqrt_sub_var_mean_4 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_div_mul_relu_sqrt_sub_var_mean_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_div_mul_relu_sqrt_sub_var_mean_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_div_mul_relu_sqrt_sub_var_mean_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = xindex // 7
    x2 = xindex
    tmp0 = (-3) + x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-3) + x0 + 4*x1), tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((-3) + x0), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.load(in_ptr0 + ((-3) + x0), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 - tmp4
    tmp8 = tmp7 * tmp7
    tmp9 = tl.load(in_ptr0 + (1 + x0), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp9 - tmp4
    tmp11 = tmp10 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = 2.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tmp5 / tmp17
    tmp19 = tl.load(in_ptr2 + (x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.load(in_ptr3 + (x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l5/cl5kcopqcnx4k2ksflkge4cjrxvsl3rinnzwrb5bumalmeqdaxsv.py
# Topologically Sorted Source Nodes: [var_mean_2, sub_2, add_4, sqrt_2, x_4, mul_2, add_5, y_3], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu]
# Source node to ATen node mapping:
#   add_4 => add_4
#   add_5 => add_5
#   mul_2 => mul_2
#   sqrt_2 => sqrt_2
#   sub_2 => sub_2
#   var_mean_2 => var_mean_2
#   x_4 => div_2
#   y_3 => relu_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_1, [1]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %getitem_5), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %sqrt_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %primals_10), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_11), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_5 = async_compile.triton('triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp3 - tmp1
    tmp5 = tmp4 * tmp4
    tmp7 = tmp6 - tmp1
    tmp8 = tmp7 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = 2.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tmp2 / tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hn/chnghwyguhzln4o437kz6yjmqlq7byenvfmbeehen7odd54b2hc7.py
# Topologically Sorted Source Nodes: [y_4, add_6], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_6
#   y_4 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_12, %primals_13, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %primals_1), kwargs = {})
triton_poi_fused_add_convolution_6 = async_compile.triton('triton_poi_fused_add_convolution_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (1, 4, 1), (4, 1, 1))
    assert_size_stride(primals_3, (1, 4, 1), (4, 1, 1))
    assert_size_stride(primals_4, (2, 4, 1), (4, 1, 1))
    assert_size_stride(primals_5, (2, ), (1, ))
    assert_size_stride(primals_6, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_7, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_8, (2, 2, 4), (8, 4, 1))
    assert_size_stride(primals_9, (2, ), (1, ))
    assert_size_stride(primals_10, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_11, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_12, (4, 2, 1), (2, 1, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf1 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean, add, sqrt], Original ATen: [aten.var_mean, aten.add, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sqrt_var_mean_0.run(primals_1, buf0, buf1, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean, sub, add, sqrt, x, mul, add_1, y], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_1.run(primals_1, buf0, buf1, primals_2, primals_3, buf2, 16, grid=grid(16), stream=stream0)
        del buf0
        del primals_2
        del primals_3
        # Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (1, 2, 4), (8, 4, 1))
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf4, primals_5, 8, grid=grid(8), stream=stream0)
        del primals_5
        buf5 = reinterpret_tensor(buf1, (1, 1, 4), (4, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [var_mean_1], Original ATen: [aten.var_mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_var_mean_3.run(buf4, buf5, 4, grid=grid(4), stream=stream0)
        buf6 = empty_strided_cuda((1, 2, 7), (14, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_1, sub_1, add_2, sqrt_1, x_1, mul_1, add_3, y_2, x_2], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_div_mul_relu_sqrt_sub_var_mean_4.run(buf4, buf5, primals_6, primals_7, buf6, 14, grid=grid(14), stream=stream0)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_8, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf7, (1, 2, 4), (8, 4, 1))
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf8, primals_9, 8, grid=grid(8), stream=stream0)
        del primals_9
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [var_mean_2], Original ATen: [aten.var_mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_var_mean_3.run(buf8, buf9, 4, grid=grid(4), stream=stream0)
        buf10 = empty_strided_cuda((1, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_2, sub_2, add_4, sqrt_2, x_4, mul_2, add_5, y_3], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_relu_sqrt_sub_var_mean_5.run(buf8, buf9, primals_10, primals_11, buf10, 8, grid=grid(8), stream=stream0)
        del buf9
        del primals_11
        # Topologically Sorted Source Nodes: [y_4], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_12, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf11, (1, 4, 4), (16, 4, 1))
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [y_4, add_6], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_6.run(buf12, primals_13, primals_1, 16, grid=grid(16), stream=stream0)
        del primals_13
    return (buf12, primals_1, primals_4, primals_6, primals_7, primals_8, primals_10, primals_12, buf2, buf4, buf6, buf8, buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2, 2, 4), (8, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
