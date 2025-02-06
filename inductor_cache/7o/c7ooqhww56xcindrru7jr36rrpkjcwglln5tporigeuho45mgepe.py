# AOT ID: ['14_forward']
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


# kernel path: inductor_cache/ko/ckohw3eml5em3yli6g33lgjackbk5oyoujslduqtpdsw3avtm4dz.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_1 => div, mul_1, pow_1, pow_2, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_3, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_2, %pow_2), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %div), kwargs = {})
triton_per_fused__weight_norm_interface_0 = async_compile.triton('triton_per_fused__weight_norm_interface_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 12*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 12*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ib/cibyvysfg67jidxkm5qzvtwywzt7uxdk23kokkzuhxnyqs5gehsu.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   x => gt, mul, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, 0.1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %primals_1, %mul), kwargs = {})
triton_poi_fused_leaky_relu_1 = async_compile.triton('triton_poi_fused_leaky_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3n/c3nxetlbytqu27tc2l4m7yck3r4yhskmckshmcjztthiziyhnynh.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   x_2 => gt_1, mul_2, where_1
# Graph fragment:
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%squeeze, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 0.1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %squeeze, %mul_2), kwargs = {})
#   %gt_10 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%squeeze_3, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_2 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7 > tmp3
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h5/ch5562zk542ihqxhorlh6ouy5fisxcgzru7p3biq4ylfadab5o7r.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   x_6 => gt_2, mul_4, where_2
# Graph fragment:
#   %gt_2 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%squeeze_5, 0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 0.1), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %squeeze_5, %mul_4), kwargs = {})
triton_poi_fused_leaky_relu_3 = async_compile.triton('triton_poi_fused_leaky_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.1
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvujn4qhsbvrl3k7coaxzmu6zirh444svfgfroxvwri5w5jwpyz.py
# Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_11 => add_1
#   x_12 => gt_4, mul_8, where_4
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_11, %squeeze_5), kwargs = {})
#   %gt_4 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%squeeze_12, 0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_12, 0.1), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %squeeze_12, %mul_8), kwargs = {})
triton_poi_fused_add_leaky_relu_4 = async_compile.triton('triton_poi_fused_add_leaky_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp11 = 0.1
    tmp12 = tmp8 * tmp11
    tmp13 = tl.where(tmp10, tmp8, tmp12)
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ck/cckflx6d2fxzfx5bxarka4asbkczfoxtq6jfnd5fyjmuy4o4h7ry.py
# Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_17 => add_2
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_18, %squeeze_12), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_poi_fused_add_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_3, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_6, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_9, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_12, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_15, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_18, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 1, 1), (1, 1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((4, 4, 3), (12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf1, primals_3, primals_2, buf2, 4, 12, grid=grid(4), stream=stream0)
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_1.run(primals_1, buf3, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(reinterpret_tensor(buf3, (1, 4, 4), (0, 4, 1), 0), buf2, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf4, (1, 4, 4), (16, 4, 1))
        buf5 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf6 = reinterpret_tensor(buf5, (4, 1, 1), (1, 1, 1), 0); del buf5  # reuse
        buf7 = empty_strided_cuda((4, 4, 3), (12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf6, primals_6, primals_5, buf7, 4, 12, grid=grid(4), stream=stream0)
        buf8 = reinterpret_tensor(buf4, (4, 4), (4, 1), 0); del buf4  # reuse
        buf36 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_2.run(buf8, primals_4, buf36, 16, grid=grid(16), stream=stream0)
        del primals_4
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(reinterpret_tensor(buf8, (1, 4, 4), (0, 4, 1), 0), buf7, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf9, (1, 4, 4), (16, 4, 1))
        buf10 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf14 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_3.run(buf9, primals_7, primals_1, buf10, buf14, 16, grid=grid(16), stream=stream0)
        buf11 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf12 = reinterpret_tensor(buf11, (4, 1, 1), (1, 1, 1), 0); del buf11  # reuse
        buf13 = empty_strided_cuda((4, 4, 3), (12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf12, primals_9, primals_8, buf13, 4, 12, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_2], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(reinterpret_tensor(buf14, (1, 4, 4), (0, 4, 1), 0), buf13, stride=(1,), padding=(3,), dilation=(3,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf15, (1, 4, 4), (16, 4, 1))
        buf16 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf17 = reinterpret_tensor(buf16, (4, 1, 1), (1, 1, 1), 0); del buf16  # reuse
        buf18 = empty_strided_cuda((4, 4, 3), (12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf17, primals_12, primals_11, buf18, 4, 12, grid=grid(4), stream=stream0)
        buf19 = reinterpret_tensor(buf15, (4, 4), (4, 1), 0); del buf15  # reuse
        buf35 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_2.run(buf19, primals_10, buf35, 16, grid=grid(16), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(reinterpret_tensor(buf19, (1, 4, 4), (0, 4, 1), 0), buf18, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf20, (1, 4, 4), (16, 4, 1))
        buf21 = reinterpret_tensor(buf20, (4, 4), (4, 1), 0); del buf20  # reuse
        buf22 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf26 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_4.run(buf21, primals_13, buf9, primals_7, primals_1, buf22, buf26, 16, grid=grid(16), stream=stream0)
        del buf9
        del primals_1
        del primals_13
        del primals_7
        buf23 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf24 = reinterpret_tensor(buf23, (4, 1, 1), (1, 1, 1), 0); del buf23  # reuse
        buf25 = empty_strided_cuda((4, 4, 3), (12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf24, primals_15, primals_14, buf25, 4, 12, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_4], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(reinterpret_tensor(buf26, (1, 4, 4), (0, 4, 1), 0), buf25, stride=(1,), padding=(5,), dilation=(5,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf27, (1, 4, 4), (16, 4, 1))
        buf28 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf29 = reinterpret_tensor(buf28, (4, 1, 1), (1, 1, 1), 0); del buf28  # reuse
        buf30 = empty_strided_cuda((4, 4, 3), (12, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf29, primals_18, primals_17, buf30, 4, 12, grid=grid(4), stream=stream0)
        buf31 = reinterpret_tensor(buf27, (4, 4), (4, 1), 0); del buf27  # reuse
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_2.run(buf31, primals_16, buf34, 16, grid=grid(16), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(reinterpret_tensor(buf31, (1, 4, 4), (0, 4, 1), 0), buf30, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf32, (1, 4, 4), (16, 4, 1))
        buf33 = reinterpret_tensor(buf32, (4, 4), (4, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf33, primals_19, buf21, 16, grid=grid(16), stream=stream0)
        del buf21
        del primals_19
    return (buf33, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, buf1, buf2, reinterpret_tensor(buf3, (1, 4, 4), (16, 4, 1), 0), buf6, buf7, reinterpret_tensor(buf8, (1, 4, 4), (16, 4, 1), 0), buf10, buf12, buf13, reinterpret_tensor(buf14, (1, 4, 4), (16, 4, 1), 0), buf17, buf18, reinterpret_tensor(buf19, (1, 4, 4), (16, 4, 1), 0), buf22, buf24, buf25, reinterpret_tensor(buf26, (1, 4, 4), (16, 4, 1), 0), buf29, buf30, reinterpret_tensor(buf31, (1, 4, 4), (16, 4, 1), 0), buf34, buf35, buf36, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
