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


# kernel path: inductor_cache/b4/cb46fre3ckpetiqcbkvlfdyohuywhozoc6aujtwvaarkqeqmleja.py
# Topologically Sorted Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_poi_fused_constant_pad_nd_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
    tl.store(out_ptr0 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ra/crambzxi5tzqgldqdrrzhrwcbl6zwg4khlwvnqu2urctt7cmgjik.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add, rsqrt, var_mean
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_1(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 16*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 16.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp23, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/27/c27qpi4bpe6bnpbtmsuwf2nntqablgrhr62qa25t5nhii23a4yig.py
# Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_1 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_3, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_2 = async_compile.triton('triton_poi_fused_constant_pad_nd_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + (x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp10, tmp17, tmp18)
    tl.store(out_ptr0 + (x4), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aw/cawfbdwz4zk5g3m2zu5qkntbyoy2nc3xg3x4busi3yst7az47l7d.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_3 => convolution_1
#   x_4 => add_1, rsqrt_1, var_mean_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_3 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_3(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 16*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 16.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.store(out_ptr2 + (x3), tmp23, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3h/c3h437ofom5l6qrgpaxf2m7763wxpgykyhyw3fm7rwmnsluw5323.py
# Topologically Sorted Source Nodes: [out, pad_2], Original ATen: [aten.add, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out => add_2
#   pad_2 => constant_pad_nd_2
# Graph fragment:
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %primals_1), kwargs = {})
#   %constant_pad_nd_2 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_2, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_4 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr2 + (x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 16.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr3 + ((-5) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tl.store(out_ptr0 + (x4), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6z/c6znozbfv6lgti2nknqes4in4opiqmoksmtbv5mrc3l7ms35xjrs.py
# Topologically Sorted Source Nodes: [out, x_8, x_9, out_1], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   out => add_2
#   out_1 => add_5
#   x_8 => convolution_3
#   x_9 => add_4, rsqrt_3, var_mean_3
# Graph fragment:
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %primals_1), kwargs = {})
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_3, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_15, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_16, %add_2), kwargs = {})
triton_per_fused__native_batch_norm_legit_add_convolution_5 = async_compile.triton('triton_per_fused__native_batch_norm_legit_add_convolution_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_add_convolution_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_add_convolution_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 16*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (r2 + 16*x3), xmask, other=0.0)
    tmp27 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (r2 + 16*x3), xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 16.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp28 = tmp26 - tmp27
    tmp30 = tmp29 / tmp20
    tmp31 = tmp30 + tmp22
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp28 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp25 + tmp35
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.store(out_ptr2 + (r2 + 16*x3), tmp36, xmask)
    tl.store(out_ptr3 + (x3), tmp24, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfwfp2xfuff4g6bbhrggqvtinnykef3uct3yw7yi3ovwgi5hsv2u.py
# Topologically Sorted Source Nodes: [out_2, x_18, x_19, out_3], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   out_2 => add_8
#   out_3 => add_11
#   x_18 => convolution_7
#   x_19 => add_10, rsqrt_7, var_mean_7
# Graph fragment:
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_26, %add_5), kwargs = {})
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %primals_16, %primals_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_35, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %add_8), kwargs = {})
triton_per_fused__native_batch_norm_legit_add_convolution_6 = async_compile.triton('triton_per_fused__native_batch_norm_legit_add_convolution_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_add_convolution_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_add_convolution_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 16*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (r2 + 16*x3), xmask, other=0.0)
    tmp27 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_out_ptr1 + (r2 + 16*x3), xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 16.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp28 = tmp26 - tmp27
    tmp30 = tmp29 / tmp20
    tmp31 = tmp30 + tmp22
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp28 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp25 + tmp35
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (r2 + 16*x3), tmp36, xmask)
    tl.store(out_ptr2 + (x3), tmp24, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(primals_1, buf0, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 4, 4), (64, 16, 4, 1))
        buf2 = buf1; del buf1  # reuse
        buf3 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf6 = reinterpret_tensor(buf4, (1, 16, 1, 1), (16, 1, 1, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_1.run(buf2, buf6, primals_3, buf3, 16, 16, grid=grid(16), stream=stream0)
        del primals_3
        buf7 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2.run(buf2, buf3, buf6, buf7, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 4, 4, 4), (64, 16, 4, 1))
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf11 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf13 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_3.run(buf9, primals_5, buf10, buf11, buf13, 16, 16, grid=grid(16), stream=stream0)
        del primals_5
        buf14 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out, pad_2], Original ATen: [aten.add, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_4.run(buf9, buf10, buf11, primals_1, buf14, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 4, 4), (64, 16, 4, 1))
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf18 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf20 = reinterpret_tensor(buf18, (1, 16, 1, 1), (16, 1, 1, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_1.run(buf16, buf20, primals_7, buf17, 16, 16, grid=grid(16), stream=stream0)
        del primals_7
        buf21 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2.run(buf16, buf17, buf20, buf21, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 4, 4, 4), (64, 16, 4, 1))
        buf23 = buf22; del buf22  # reuse
        buf24 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf28 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf27 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [out, x_8, x_9, out_1], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_add_convolution_5.run(buf23, primals_9, buf9, buf10, buf11, primals_1, buf24, buf28, buf27, 16, 16, grid=grid(16), stream=stream0)
        del primals_1
        del primals_9
        buf29 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_4], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(buf28, buf29, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 4, 4, 4), (64, 16, 4, 1))
        buf31 = buf30; del buf30  # reuse
        buf32 = reinterpret_tensor(buf11, (1, 16, 1, 1), (16, 1, 1, 1), 0); del buf11  # reuse
        buf33 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf35 = reinterpret_tensor(buf33, (1, 16, 1, 1), (16, 1, 1, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_10, x_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_1.run(buf31, buf35, primals_11, buf32, 16, 16, grid=grid(16), stream=stream0)
        del primals_11
        buf36 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_5], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2.run(buf31, buf32, buf35, buf36, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 4, 4, 4), (64, 16, 4, 1))
        buf38 = buf37; del buf37  # reuse
        buf39 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf40 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf42 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_3.run(buf38, primals_13, buf39, buf40, buf42, 16, 16, grid=grid(16), stream=stream0)
        del primals_13
        buf43 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2, pad_6], Original ATen: [aten.add, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_4.run(buf38, buf39, buf40, buf28, buf43, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 4, 4, 4), (64, 16, 4, 1))
        buf45 = buf44; del buf44  # reuse
        buf46 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf47 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf49 = reinterpret_tensor(buf47, (1, 16, 1, 1), (16, 1, 1, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_15, x_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_1.run(buf45, buf49, primals_15, buf46, 16, 16, grid=grid(16), stream=stream0)
        del primals_15
        buf50 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_7], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2.run(buf45, buf46, buf49, buf50, 576, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 4, 4, 4), (64, 16, 4, 1))
        buf52 = buf51; del buf51  # reuse
        buf53 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        buf57 = buf28; del buf28  # reuse
        buf56 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [out_2, x_18, x_19, out_3], Original ATen: [aten.add, aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_add_convolution_6.run(buf52, buf57, primals_17, buf38, buf39, buf40, buf53, buf56, 16, 16, grid=grid(16), stream=stream0)
        del buf40
        del primals_17
    return (buf57, primals_2, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, buf0, buf2, buf3, buf6, buf7, buf9, reinterpret_tensor(buf13, (16, ), (1, ), 0), buf14, buf16, buf17, buf20, buf21, buf23, reinterpret_tensor(buf27, (16, ), (1, ), 0), buf29, buf31, buf32, buf35, buf36, buf38, reinterpret_tensor(buf42, (16, ), (1, ), 0), buf43, buf45, buf46, buf49, buf50, buf52, reinterpret_tensor(buf56, (16, ), (1, ), 0), reinterpret_tensor(buf53, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf39, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
