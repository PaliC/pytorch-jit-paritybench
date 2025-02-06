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


# kernel path: inductor_cache/no/cnorc6sepsnbwc5flo2cuw7vz4rruy55gwmmo3quyaguqpybqtph.py
# Topologically Sorted Source Nodes: [x, x_1, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.cat]
# Source node to ATen node mapping:
#   out => cat
#   x => convolution
#   x_1 => add, rsqrt, var_mean
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_1, %view_8, %view_15, %view_22], 1), kwargs = {})
triton_per_fused__native_batch_norm_legit_cat_convolution_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_cat_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_cat_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_cat_convolution_0(in_out_ptr0, in_ptr0, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    x0 = (xindex % 2)
    x1 = xindex // 2
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
    tmp19 = tmp2 - tmp12
    tmp20 = 16.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.store(out_ptr2 + (r2 + 16*x0 + 128*x1), tmp25, xmask)
    tl.store(out_ptr3 + (x3), tmp24, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sb/csb4jfo3hgmm5wqzc36kle67aevyneuw6nlptnvlqlacl5crxirq.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_2 => convolution_1
#   x_3 => add_1, rsqrt_1, var_mean_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_1(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    x0 = (xindex % 2)
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


# kernel path: inductor_cache/sg/csgi6c3vjiv3gxczficd5gglw6272l4eg6tup22ndltq6avytlu2.py
# Topologically Sorted Source Nodes: [pad_2], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_2 => constant_pad_nd_2
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_5, [3, 3, 3, 3], 0.0), kwargs = {})
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
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 10) % 10)
    x0 = (xindex % 10)
    x2 = xindex // 100
    x4 = xindex
    tmp0 = (-3) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-15) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
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


# kernel path: inductor_cache/pc/cpcrxhvdunyngfiqo2yngyl2i7p3bu7h62jf2fyrfxsuggvkfw6a.py
# Topologically Sorted Source Nodes: [pad_4], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_4 => constant_pad_nd_4
# Graph fragment:
#   %constant_pad_nd_4 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_12, [6, 6, 6, 6], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_3 = async_compile.triton('triton_poi_fused_constant_pad_nd_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = (-6) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-6) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-30) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
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


# kernel path: inductor_cache/4s/c4syba2btgigxlvesfpo4pv3kwrcnsvfyck6spmovt4sv5miscrr.py
# Topologically Sorted Source Nodes: [pad_6], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_6 => constant_pad_nd_6
# Graph fragment:
#   %constant_pad_nd_6 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_19, [9, 9, 9, 9], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_4 = async_compile.triton('triton_poi_fused_constant_pad_nd_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 22) % 22)
    x0 = (xindex % 22)
    x2 = xindex // 484
    x4 = xindex
    tmp0 = (-9) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-9) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-45) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
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


# kernel path: inductor_cache/p2/cp2abd4gflqjnzojbpsaecbbwx5meiyvi75g4iz7cg6a3sfgjyuc.py
# Topologically Sorted Source Nodes: [x_17, x_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_17 => convolution_7
#   x_18 => add_7, mul_7, rsqrt_7, sub_7, var_mean_7
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_16, %primals_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_23, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_23, %getitem_15), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_5 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_5(in_out_ptr0, in_ptr0, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    x0 = (xindex % 8)
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
    tmp19 = tmp2 - tmp12
    tmp20 = 16.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.store(out_ptr2 + (r2 + 16*x3), tmp25, xmask)
    tl.store(out_ptr3 + (x3), tmp24, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (2, ), (1, ))
    assert_size_stride(primals_4, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_5, (2, ), (1, ))
    assert_size_stride(primals_6, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_7, (2, ), (1, ))
    assert_size_stride(primals_8, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_9, (2, ), (1, ))
    assert_size_stride(primals_10, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_11, (2, ), (1, ))
    assert_size_stride(primals_12, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_13, (2, ), (1, ))
    assert_size_stride(primals_14, (2, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_15, (2, ), (1, ))
    assert_size_stride(primals_16, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_17, (8, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 2, 4, 4), (32, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf49 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        buf45 = reinterpret_tensor(buf49, (4, 2, 4, 4), (128, 16, 4, 1), 0)  # alias
        buf5 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.cat]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_cat_convolution_0.run(buf1, primals_3, buf2, buf45, buf5, 8, 16, grid=grid(8), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(primals_1, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 2, 4, 4), (32, 16, 4, 1))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf11 = reinterpret_tensor(buf9, (1, 8, 1, 1), (8, 1, 1, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_1.run(buf7, buf11, primals_5, buf8, 8, 16, grid=grid(8), stream=stream0)
        del primals_5
        buf12 = empty_strided_cuda((4, 2, 10, 10), (200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_2], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2.run(buf7, buf8, buf11, buf12, 800, grid=grid(800), stream=stream0)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_6, stride=(1, 1), padding=(0, 0), dilation=(3, 3), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 2, 4, 4), (32, 16, 4, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf46 = reinterpret_tensor(buf49, (4, 2, 4, 4), (128, 16, 4, 1), 32)  # alias
        buf18 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.cat]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_cat_convolution_0.run(buf14, primals_7, buf15, buf46, buf18, 8, 16, grid=grid(8), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(primals_1, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 2, 4, 4), (32, 16, 4, 1))
        buf20 = buf19; del buf19  # reuse
        buf21 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        buf22 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf24 = reinterpret_tensor(buf22, (1, 8, 1, 1), (8, 1, 1, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_1.run(buf20, buf24, primals_9, buf21, 8, 16, grid=grid(8), stream=stream0)
        del primals_9
        buf25 = empty_strided_cuda((4, 2, 16, 16), (512, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_4], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_3.run(buf20, buf21, buf24, buf25, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_10, stride=(1, 1), padding=(0, 0), dilation=(6, 6), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 2, 4, 4), (32, 16, 4, 1))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf47 = reinterpret_tensor(buf49, (4, 2, 4, 4), (128, 16, 4, 1), 64)  # alias
        buf31 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_10, x_11, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.cat]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_cat_convolution_0.run(buf27, primals_11, buf28, buf47, buf31, 8, 16, grid=grid(8), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(primals_1, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 2, 4, 4), (32, 16, 4, 1))
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        buf35 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf37 = reinterpret_tensor(buf35, (1, 8, 1, 1), (8, 1, 1, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_1.run(buf33, buf37, primals_13, buf34, 8, 16, grid=grid(8), stream=stream0)
        del primals_13
        buf38 = empty_strided_cuda((4, 2, 22, 22), (968, 484, 22, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_6], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_4.run(buf33, buf34, buf37, buf38, 3872, grid=grid(3872), stream=stream0)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_14, stride=(1, 1), padding=(0, 0), dilation=(9, 9), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 2, 4, 4), (32, 16, 4, 1))
        buf40 = buf39; del buf39  # reuse
        buf41 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf48 = reinterpret_tensor(buf49, (4, 2, 4, 4), (128, 16, 4, 1), 96)  # alias
        buf44 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_15, x_16, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.cat]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_cat_convolution_0.run(buf40, primals_15, buf41, buf48, buf44, 8, 16, grid=grid(8), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 8, 4, 4), (128, 16, 4, 1))
        buf51 = buf50; del buf50  # reuse
        buf52 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        buf56 = empty_strided_cuda((1, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        buf55 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_17, x_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_5.run(buf51, primals_17, buf52, buf56, buf55, 32, 16, grid=grid(32), stream=stream0)
        del primals_17
    return (reinterpret_tensor(buf56, (4, 8, 4, 4), (128, 16, 4, 1), 0), primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, buf1, reinterpret_tensor(buf5, (8, ), (1, ), 0), buf7, buf8, buf11, buf12, buf14, reinterpret_tensor(buf18, (8, ), (1, ), 0), buf20, buf21, buf24, buf25, buf27, reinterpret_tensor(buf31, (8, ), (1, ), 0), buf33, buf34, buf37, buf38, buf40, reinterpret_tensor(buf44, (8, ), (1, ), 0), buf49, buf51, reinterpret_tensor(buf55, (32, ), (1, ), 0), reinterpret_tensor(buf52, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf41, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf28, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf15, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf2, (1, 8, 1, 1), (8, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
