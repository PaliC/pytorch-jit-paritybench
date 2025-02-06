# AOT ID: ['76_forward']
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


# kernel path: inductor_cache/nu/cnuppexjethwgt4jnhe6llgn4fzgkulav4lwmz7e3n7u2uwcxwpe.py
# Topologically Sorted Source Nodes: [x_right], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_right => _low_memory_max_pool2d_with_offsets, getitem_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%convolution, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex // 16
    x3 = xindex // 4096
    x6 = (xindex % 4096)
    x7 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x0 + 64*x5), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x0 + 64*x5), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x0 + 64*x5), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x5), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 64*x5), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x5), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x0 + 64*x5), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x5), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x5), tmp49, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tl.store(out_ptr0 + (x6 + 8192*x3), tmp51, None)
    tl.store(out_ptr1 + (x7), tmp76, None)
''', device_str='cuda')


# kernel path: inductor_cache/mo/cmok3ypryhlnwultdrwhukfbqhmqu7deqozucj6tvcu3slkcveag.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_2, %getitem], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/tq/ctqai5jqkfivocjjzhreptmhv6p54z4qzot5o6d7fslnqfxqeox2.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_2 => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_9), kwargs = {})
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/uy/cuydgikis3xpcijtzvlof6yayndkj7ajtdiz5dn3tnmdh2phv2ps.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_6 => add_2
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_16, %convolution_18), kwargs = {})
triton_poi_fused_add_3 = async_compile.triton('triton_poi_fused_add_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/e4/ce47wwzzuz6whzn4hzlcnllhq355bwgcvmwlq4ohdozi3psdob4b.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_10 => add_4
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convolution_27), kwargs = {})
triton_poi_fused_add_4 = async_compile.triton('triton_poi_fused_add_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gt/cgttotvorkpmfdu5jeoyxxvrcvp5nfplh3pgryokxmbfepjlpzfw.py
# Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   input_32 => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_7, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_5 = async_compile.triton('triton_poi_fused_mean_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6j/c6j7ype6bhmbkdb7c73543djjfgiarl5nrsghld6tr6nzrb4gpth.py
# Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_14 => add_8
# Graph fragment:
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %convolution_37), kwargs = {})
triton_poi_fused_add_6 = async_compile.triton('triton_poi_fused_add_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_4, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_5, (16, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_6, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_7, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_8, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_9, (32, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_10, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_12, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_13, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_15, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_16, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_17, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_18, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_19, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_20, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_21, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_22, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_24, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_26, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_28, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_29, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_30, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_31, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_33, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_34, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_35, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_36, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_37, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_38, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_39, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_40, (128, 128, 3, 3), (1152, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 32, 32), (16384, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf6 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        buf3 = reinterpret_tensor(buf6, (4, 16, 16, 16), (8192, 256, 16, 1), 4096)  # alias
        buf4 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_right], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0.run(buf0, buf3, buf4, 16384, grid=grid(16384), stream=stream0)
        buf5 = reinterpret_tensor(buf6, (4, 16, 16, 16), (8192, 256, 16, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf2, buf5, 16384, grid=grid(16384), stream=stream0)
        del buf2
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf9, (4, 96, 8, 8), (6144, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf10, (4, 96, 8, 8), (6144, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf7, primals_10, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf12, (4, 16, 8, 8), (1024, 64, 8, 1))
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf14 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf14, buf13, 8192, grid=grid(8192), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf16, (4, 192, 8, 8), (12288, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf18, buf14, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_16, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf20, (4, 192, 4, 4), (3072, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf21, (4, 192, 4, 4), (3072, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf18, primals_19, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf23, (4, 32, 4, 4), (512, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf25 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_3.run(buf25, buf24, 4096, grid=grid(4096), stream=stream0)
        del buf24
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf27, (4, 384, 4, 4), (6144, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_3.run(buf29, buf25, 4096, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf31, (4, 384, 2, 2), (1536, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf32, (4, 384, 2, 2), (1536, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 128, 2, 2), (512, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf29, primals_28, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf34, (4, 64, 2, 2), (256, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 128, 2, 2), (512, 4, 2, 1))
        buf36 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_4.run(buf36, buf35, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 128, 2, 2), (512, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf38, (4, 768, 2, 2), (3072, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 128, 2, 2), (512, 4, 2, 1))
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_4.run(buf40, buf36, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 128, 2, 2), (512, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf42, (4, 768, 2, 2), (3072, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 128, 2, 2), (512, 4, 2, 1))
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_4.run(buf44, buf40, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 128, 2, 2), (512, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf46, (4, 768, 2, 2), (3072, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 128, 2, 2), (512, 4, 2, 1))
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_4.run(buf48, buf44, 2048, grid=grid(2048), stream=stream0)
        buf49 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_5.run(buf48, buf49, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_39, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 128, 1, 1), (128, 1, 1, 1))
        buf51 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf48, buf50, buf51, 2048, grid=grid(2048), stream=stream0)
        del buf50
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 128, 2, 2), (512, 4, 2, 1))
    return (buf7, buf18, buf29, buf48, buf52, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, buf0, buf1, buf4, buf6, buf7, buf8, buf9, buf10, buf12, buf14, buf15, buf16, buf18, buf19, buf20, buf21, buf23, buf25, buf26, buf27, buf29, buf30, buf31, buf32, buf34, buf36, buf37, buf38, buf40, buf41, buf42, buf44, buf45, buf46, buf49, buf51, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
