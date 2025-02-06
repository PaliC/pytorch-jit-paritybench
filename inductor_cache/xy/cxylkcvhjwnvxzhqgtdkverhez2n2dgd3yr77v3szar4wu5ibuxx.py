# AOT ID: ['11_forward']
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


# kernel path: inductor_cache/ul/culr5fdpvjyjfivll4kidi2g2gv63hljt5chlwqkdt5lfl3gwh5s.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_1 => gt, mul, where
#   x_2 => add
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %convolution), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %slice_1), kwargs = {})
triton_poi_fused__prelu_kernel_add_0 = async_compile.triton('triton_poi_fused__prelu_kernel_add_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 62) % 32)
    x0 = (xindex % 62)
    x4 = xindex // 62
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 64*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6p/c6pxxi22f53564fx47fhrbfcrrh4z6vhibi22ukx7h7zsbxjzhxn.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_4 => gt_1, mul_1, where_1
#   x_5 => add_1
# Graph fragment:
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %convolution_2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_2, %mul_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_1, %slice_2), kwargs = {})
triton_poi_fused__prelu_kernel_add_1 = async_compile.triton('triton_poi_fused__prelu_kernel_add_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 60) % 32)
    x0 = (xindex % 60)
    x4 = xindex // 60
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 62*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7w/c7wnagkpr6xlsm55bhvxiyr7we3itnvk7umv247qkd4qdiv4coxd.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_7 => gt_2, mul_2, where_2
#   x_8 => add_2
# Graph fragment:
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_4, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %convolution_4), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_4, %mul_2), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_2, %slice_3), kwargs = {})
triton_poi_fused__prelu_kernel_add_2 = async_compile.triton('triton_poi_fused__prelu_kernel_add_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 58) % 32)
    x0 = (xindex % 58)
    x4 = xindex // 58
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 60*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fr/cfrsy336auomo6wsxafws3io5ltz2rayxi6gbhddfmxqxo5k5fgx.py
# Topologically Sorted Source Nodes: [x_10, x_11], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_10 => gt_3, mul_3, where_3
#   x_11 => add_3
# Graph fragment:
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_6, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %convolution_6), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_6, %mul_3), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_3, %slice_4), kwargs = {})
triton_poi_fused__prelu_kernel_add_3 = async_compile.triton('triton_poi_fused__prelu_kernel_add_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 56) % 32)
    x0 = (xindex % 56)
    x4 = xindex // 56
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 58*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4i/c4iwaq3mn6ypmneffhlmhajqvhh7z73ptykxcxdnch6katlzg42r.py
# Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_13 => gt_4, mul_4, where_4
#   x_14 => add_4
# Graph fragment:
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_8, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %convolution_8), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %convolution_8, %mul_4), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_4, %slice_5), kwargs = {})
triton_poi_fused__prelu_kernel_add_4 = async_compile.triton('triton_poi_fused__prelu_kernel_add_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 54) % 32)
    x0 = (xindex % 54)
    x4 = xindex // 54
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 56*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7z/c7zljqxc37yqybv5ogonwaxk576lykxzgibl6n2rchszcbzlqlod.py
# Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_16 => gt_5, mul_5, where_5
#   x_17 => add_5
# Graph fragment:
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_10, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %convolution_10), kwargs = {})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %convolution_10, %mul_5), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_5, %slice_6), kwargs = {})
triton_poi_fused__prelu_kernel_add_5 = async_compile.triton('triton_poi_fused__prelu_kernel_add_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 52) % 32)
    x0 = (xindex % 52)
    x4 = xindex // 52
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 54*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sf/csfirnw3tsewkjpwikpbqh5m5bbkwzzipa33ysnx7flv577svv6x.py
# Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_19 => gt_6, mul_6, where_6
#   x_20 => add_6
# Graph fragment:
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_12, 0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %convolution_12), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %convolution_12, %mul_6), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_6, %slice_7), kwargs = {})
triton_poi_fused__prelu_kernel_add_6 = async_compile.triton('triton_poi_fused__prelu_kernel_add_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 50) % 32)
    x0 = (xindex % 50)
    x4 = xindex // 50
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 52*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h2/ch2uxwnhtmhsmqrsxnwfphi2m6popu6qczdy64rpm64rz7qwh2eb.py
# Topologically Sorted Source Nodes: [x_22, x_23], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_22 => gt_7, mul_7, where_7
#   x_23 => add_7
# Graph fragment:
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_14, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %convolution_14), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %convolution_14, %mul_7), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_7, %slice_8), kwargs = {})
triton_poi_fused__prelu_kernel_add_7 = async_compile.triton('triton_poi_fused__prelu_kernel_add_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 48) % 32)
    x0 = (xindex % 48)
    x4 = xindex // 48
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 50*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pr/cprf4kltncgzyfnfz2qreyzvvgn2rmxlzvecm2sbu36ds3qunvkt.py
# Topologically Sorted Source Nodes: [x_25, x_26], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_25 => gt_8, mul_8, where_8
#   x_26 => add_8
# Graph fragment:
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_16, 0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %convolution_16), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %convolution_16, %mul_8), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_8, %slice_9), kwargs = {})
triton_poi_fused__prelu_kernel_add_8 = async_compile.triton('triton_poi_fused__prelu_kernel_add_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 46) % 32)
    x0 = (xindex % 46)
    x4 = xindex // 46
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 48*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z5/cz5dbha4tqdi7yhtvutoyjtripwiq22xnsm6sdmzcuhslnxgv567.py
# Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten._prelu_kernel, aten.add]
# Source node to ATen node mapping:
#   x_28 => gt_9, mul_9, where_9
#   x_29 => add_9
# Graph fragment:
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_18, 0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %convolution_18), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %convolution_18, %mul_9), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_9, %slice_10), kwargs = {})
triton_poi_fused__prelu_kernel_add_9 = async_compile.triton('triton_poi_fused__prelu_kernel_add_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 44) % 32)
    x0 = (xindex % 44)
    x4 = xindex // 44
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + x0 + 46*x4), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxr7owicvgacvvlecd4ozhzamr4rutkhhicctxg7pyve3c73t3a.py
# Topologically Sorted Source Nodes: [conv1d_20, tanh], Original ATen: [aten.convolution, aten.tanh]
# Source node to ATen node mapping:
#   conv1d_20 => convolution_20
#   tanh => tanh
# Graph fragment:
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_9, %primals_32, %primals_33, [1], [0], [1], False, [0], 1), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_20,), kwargs = {})
triton_poi_fused_convolution_tanh_10 = async_compile.triton('triton_poi_fused_convolution_tanh_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_tanh_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_tanh_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = libdevice.tanh(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_2, (32, 1, 3), (3, 3, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_5, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_8, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_11, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_14, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_17, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_20, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_23, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_26, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_29, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_32, (1, 32, 1), (32, 1, 1))
    assert_size_stride(primals_33, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 62), (1984, 62, 1))
        # Topologically Sorted Source Nodes: [x_res], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_1, primals_4, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf1, (4, 32, 64), (2048, 64, 1))
        buf2 = empty_strided_cuda((4, 32, 62), (1984, 62, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_0.run(buf0, primals_3, buf1, buf2, 7936, grid=grid(7936), stream=stream0)
        del buf1
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_5, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (4, 32, 60), (1920, 60, 1))
        # Topologically Sorted Source Nodes: [x_res_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_7, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf4, (4, 32, 62), (1984, 62, 1))
        buf5 = empty_strided_cuda((4, 32, 60), (1920, 60, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_1.run(buf3, primals_6, buf4, buf5, 7680, grid=grid(7680), stream=stream0)
        del buf4
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_8, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 58), (1856, 58, 1))
        # Topologically Sorted Source Nodes: [x_res_2], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf5, primals_10, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf7, (4, 32, 60), (1920, 60, 1))
        buf8 = empty_strided_cuda((4, 32, 58), (1856, 58, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_2.run(buf6, primals_9, buf7, buf8, 7424, grid=grid(7424), stream=stream0)
        del buf7
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_11, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf9, (4, 32, 56), (1792, 56, 1))
        # Topologically Sorted Source Nodes: [x_res_3], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf8, primals_13, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf10, (4, 32, 58), (1856, 58, 1))
        buf11 = empty_strided_cuda((4, 32, 56), (1792, 56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_10, x_11], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_3.run(buf9, primals_12, buf10, buf11, 7168, grid=grid(7168), stream=stream0)
        del buf10
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_14, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf12, (4, 32, 54), (1728, 54, 1))
        # Topologically Sorted Source Nodes: [x_res_4], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf11, primals_16, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf13, (4, 32, 56), (1792, 56, 1))
        buf14 = empty_strided_cuda((4, 32, 54), (1728, 54, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_4.run(buf12, primals_15, buf13, buf14, 6912, grid=grid(6912), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_17, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf15, (4, 32, 52), (1664, 52, 1))
        # Topologically Sorted Source Nodes: [x_res_5], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf14, primals_19, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf16, (4, 32, 54), (1728, 54, 1))
        buf17 = empty_strided_cuda((4, 32, 52), (1664, 52, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_5.run(buf15, primals_18, buf16, buf17, 6656, grid=grid(6656), stream=stream0)
        del buf16
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 50), (1600, 50, 1))
        # Topologically Sorted Source Nodes: [x_res_6], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf17, primals_22, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf19, (4, 32, 52), (1664, 52, 1))
        buf20 = empty_strided_cuda((4, 32, 50), (1600, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_6.run(buf18, primals_21, buf19, buf20, 6400, grid=grid(6400), stream=stream0)
        del buf19
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_23, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf21, (4, 32, 48), (1536, 48, 1))
        # Topologically Sorted Source Nodes: [x_res_7], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf20, primals_25, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf22, (4, 32, 50), (1600, 50, 1))
        buf23 = empty_strided_cuda((4, 32, 48), (1536, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_22, x_23], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_7.run(buf21, primals_24, buf22, buf23, 6144, grid=grid(6144), stream=stream0)
        del buf22
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_26, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 46), (1472, 46, 1))
        # Topologically Sorted Source Nodes: [x_res_8], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf23, primals_28, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf25, (4, 32, 48), (1536, 48, 1))
        buf26 = empty_strided_cuda((4, 32, 46), (1472, 46, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_25, x_26], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_8.run(buf24, primals_27, buf25, buf26, 5888, grid=grid(5888), stream=stream0)
        del buf25
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_29, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf27, (4, 32, 44), (1408, 44, 1))
        # Topologically Sorted Source Nodes: [x_res_9], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf26, primals_31, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=32, bias=None)
        assert_size_stride(buf28, (4, 32, 46), (1472, 46, 1))
        buf29 = empty_strided_cuda((4, 32, 44), (1408, 44, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten._prelu_kernel, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_9.run(buf27, primals_30, buf28, buf29, 5632, grid=grid(5632), stream=stream0)
        del buf28
        # Topologically Sorted Source Nodes: [conv1d_20], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_32, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf30, (4, 1, 44), (44, 44, 1))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [conv1d_20, tanh], Original ATen: [aten.convolution, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_tanh_10.run(buf31, primals_33, 176, grid=grid(176), stream=stream0)
        del primals_33
    return (buf31, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, buf0, buf2, buf3, buf5, buf6, buf8, buf9, buf11, buf12, buf14, buf15, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf29, buf31, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
