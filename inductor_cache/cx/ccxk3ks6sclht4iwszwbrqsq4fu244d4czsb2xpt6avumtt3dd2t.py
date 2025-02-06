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


# kernel path: inductor_cache/bs/cbswspwd52ty6v4yneq72dugbszvc6kla6dzer7jvkohkk2kuvif.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => gt, mul, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
triton_poi_fused_convolution_leaky_relu_0 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
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


# kernel path: inductor_cache/2f/c2fgxp22jfhmsh6meg3vjilnrw5cfbn4ctc7mgcrymc2djbb22kt.py
# Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_3 => convolution_1
#   input_4 => add, rsqrt, var_mean
#   input_5 => gt_1, mul_2, where_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %primals_4, %primals_5, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_1, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %view_1, %mul_2), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_1(in_out_ptr0, in_ptr0, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 256*x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp2 - tmp10
    tmp17 = 256.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp23 = 0.0
    tmp24 = tmp22 > tmp23
    tmp25 = 0.2
    tmp26 = tmp22 * tmp25
    tmp27 = tl.where(tmp24, tmp22, tmp26)
    tl.store(in_out_ptr0 + (r2 + 256*x3), tmp2, None)
    tl.store(out_ptr2 + (r2 + 256*x3), tmp27, None)
    tl.store(out_ptr3 + (x3), tmp21, None)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/q3/cq3kmbax2qczkmj4xnqloep4vchccfr6rtdimu6vgqoisg4kdkz2.py
# Topologically Sorted Source Nodes: [input_6, input_7, input_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_6 => convolution_2
#   input_7 => add_1, rsqrt_1, var_mean_1
#   input_8 => gt_2, mul_4, where_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%view_3, %primals_6, %primals_7, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_6, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, 0.2), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %view_6, %mul_4), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_2 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_2(in_out_ptr0, in_ptr0, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp28 = 0.2
    tmp29 = tmp25 * tmp28
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tl.store(in_out_ptr0 + (r2 + 64*x3), tmp2, xmask)
    tl.store(out_ptr2 + (r2 + 64*x3), tmp30, xmask)
    tl.store(out_ptr3 + (x3), tmp24, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/on/con45dogmsquhfirckkmuqjrtqw572npwqpo6mhbon6fvcackmgu.py
# Topologically Sorted Source Nodes: [input_9, input_10, input_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_10 => add_2, rsqrt_2, var_mean_2
#   input_11 => gt_3, mul_6, where_3
#   input_9 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%view_8, %primals_8, %primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_10, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_11, 0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, 0.2), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %view_11, %mul_6), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_3 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_3(in_out_ptr0, in_ptr0, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 49*x3), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 49, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 49.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp28 = 0.2
    tmp29 = tmp25 * tmp28
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tl.store(in_out_ptr0 + (r2 + 49*x3), tmp2, rmask & xmask)
    tl.store(out_ptr2 + (r2 + 49*x3), tmp30, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp24, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p3/cp3x2uzqzti4w72l2pgamkmljqpqlk7cuuwzxga3xjauiogwrv6x.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_12 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_13, %primals_10, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_7, (256, ), (1, ))
    assert_size_stride(primals_8, (512, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (1, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(primals_11, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_0.run(buf1, primals_2, 262144, grid=grid(262144), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf8 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_1.run(buf3, primals_5, buf4, buf8, buf7, 512, 256, grid=grid(512), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        buf15 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7, input_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_2.run(buf10, primals_7, buf11, buf15, buf14, 1024, 64, grid=grid(1024), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 512, 7, 7), (25088, 49, 7, 1))
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf22 = empty_strided_cuda((4, 512, 7, 7), (25088, 49, 7, 1), torch.float32)
        buf21 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, input_10, input_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_leaky_relu_3.run(buf17, primals_9, buf18, buf22, buf21, 2048, 49, grid=grid(2048), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 1, 6, 6), (36, 36, 6, 1))
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_4.run(buf24, primals_11, 144, grid=grid(144), stream=stream0)
        del primals_11
    return (buf24, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, buf1, buf3, reinterpret_tensor(buf7, (512, ), (1, ), 0), buf8, buf10, reinterpret_tensor(buf14, (1024, ), (1, ), 0), buf15, buf17, reinterpret_tensor(buf21, (2048, ), (1, ), 0), buf22, reinterpret_tensor(buf18, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf11, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 512, 1, 1), (512, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
