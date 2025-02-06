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


# kernel path: inductor_cache/b6/cb6r3c4fiklf6cbcnwf62fkpyapfxkwgjmt7lsqehqmmemexj6yd.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_1 => pow_1, pow_2, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_3, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2, 3], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
triton_poi_fused__weight_norm_interface_0 = async_compile.triton('triton_poi_fused__weight_norm_interface_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 5*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 5*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 5*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4 + 5*x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbrfzrtbeihozipwozwxtlq2p4tbmumxvttjfx5io66oq4mdafmq.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_1 => div, mul
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_2, %pow_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %div), kwargs = {})
triton_poi_fused__weight_norm_interface_1 = async_compile.triton('triton_poi_fused__weight_norm_interface_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cu/ccupozr665zowk7qe4gm3d4eg62rwqk6kiclh2xtcst27jld555g.py
# Topologically Sorted Source Nodes: [conv2d, x_2], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d => convolution
#   x_2 => gt, mul_1, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view, %mul, %primals_4, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_2 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dm/cdm3uokotllxm55fezs3k32wjpumyrigr5h7cqokdkqr2cavgtbj.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_3 => div_1, mul_2, pow_3, pow_4, sum_2
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_6, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1, 2, 3], True), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_5, %pow_4), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %div_1), kwargs = {})
triton_per_fused__weight_norm_interface_3 = async_compile.triton('triton_per_fused__weight_norm_interface_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 160
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 160*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 160*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yt/cytqrntptt4vbt45jhza42hbddo364mjq36rvxjyg65diayqfb6n.py
# Topologically Sorted Source Nodes: [conv2d_1, x_4], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   x_4 => gt_1, mul_3, where_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %mul_2, %primals_7, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.1), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %mul_3), kwargs = {})
triton_poi_fused_convolution_leaky_relu_4 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mg/cmgg2hoxxz3finf5wicdorkiw4w4bc5ymidsooy7vfkn6m3ao7un.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_5 => div_2, mul_4, pow_5, pow_6, sum_3
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_9, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1, 2, 3], True), kwargs = {})
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_8, %pow_6), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, %div_2), kwargs = {})
triton_per_fused__weight_norm_interface_5 = async_compile.triton('triton_per_fused__weight_norm_interface_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 640
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 640*x0), rmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr0 + (r1 + 640*x0), tmp9, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/vd/cvdu5oxcnlk2puxwhktvsh6qltj6ffiouztjsxuygro3yikgwybr.py
# Topologically Sorted Source Nodes: [conv2d_2, x_6], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_2 => convolution_2
#   x_6 => gt_2, mul_5, where_2
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_1, %mul_4, %primals_10, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_5), kwargs = {})
triton_poi_fused_convolution_leaky_relu_6 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/ry/crynib7ucrbkuadeelq7cvydmsph7feeaqnzxc2hzjwi3gf3ar3c.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_7 => div_3, mul_6, pow_7, pow_8, sum_4
# Graph fragment:
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_12, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2, 3], True), kwargs = {})
#   %pow_8 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_11, %pow_8), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, %div_3), kwargs = {})
triton_red_fused__weight_norm_interface_7 = async_compile.triton('triton_red_fused__weight_norm_interface_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2560*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 2560*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 2560*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ly/clydswoxynvdbevc3v36zvcpks37yenz56jxtvxdz7y5ubsxohp2.py
# Topologically Sorted Source Nodes: [conv2d_3, x_8], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
#   x_8 => gt_3, mul_7, where_3
# Graph fragment:
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %mul_6, %primals_13, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 0.1), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_3, %mul_7), kwargs = {})
triton_poi_fused_convolution_leaky_relu_8 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/je/cjec2p5j5zbs4uonx3ghsjighw5irx3nt25i3fzpfkdrme56mzh7.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_9 => div_4, mul_8, pow_10, pow_9, sum_5
# Graph fragment:
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_15, 2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [1, 2, 3], True), kwargs = {})
#   %pow_10 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_14, %pow_10), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_15, %div_4), kwargs = {})
triton_red_fused__weight_norm_interface_9 = async_compile.triton('triton_red_fused__weight_norm_interface_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 5120*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 5120*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 5120*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crbaxon6p2ity2tdmlc2fr4q34k62konwicdw7awpvrotesise7h.py
# Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_11 => div_5, mul_10, pow_11, pow_12, sum_6
# Graph fragment:
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_18, 2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2, 3], True), kwargs = {})
#   %pow_12 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_6, 0.5), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_17, %pow_12), kwargs = {})
#   %mul_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, %div_5), kwargs = {})
triton_red_fused__weight_norm_interface_10 = async_compile.triton('triton_red_fused__weight_norm_interface_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_10(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp6 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp8 / tmp5
        tmp10 = tmp6 * tmp9
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/by/cbyxwbn2cy7nob67hb6oanfrx6lastzgjdlgzzyh6qvm4l2vqvc4.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_12 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_4, %mul_10, %primals_19, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 4), (4, 4, 1))
    assert_size_stride(primals_2, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_3, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_6, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_9, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_12, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_15, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    assert_size_stride(primals_17, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_18, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_19, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_0.run(primals_3, buf0, 32, grid=grid(32), stream=stream0)
        buf1 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_1.run(primals_3, primals_2, buf0, buf1, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(reinterpret_tensor(primals_1, (4, 1, 1, 4), (4, 4, 4, 1), 0), buf1, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 1, 4), (128, 4, 4, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [conv2d, x_2], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_2.run(buf3, primals_4, 512, grid=grid(512), stream=stream0)
        del primals_4
        buf4 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf5 = reinterpret_tensor(buf4, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf4  # reuse
        buf6 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf5, primals_6, primals_5, buf6, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf3, buf6, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 128, 1, 4), (512, 4, 4, 1))
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, x_4], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_4.run(buf8, primals_7, 2048, grid=grid(2048), stream=stream0)
        del primals_7
        buf9 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf10 = reinterpret_tensor(buf9, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf9  # reuse
        buf11 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_5.run(buf10, primals_9, primals_8, buf11, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf8, buf11, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 512, 1, 4), (2048, 4, 4, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [conv2d_2, x_6], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_6.run(buf13, primals_10, 8192, grid=grid(8192), stream=stream0)
        del primals_10
        buf14 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf15 = reinterpret_tensor(buf14, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf14  # reuse
        buf16 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_7.run(buf15, primals_12, primals_11, buf16, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf13, buf16, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 1024, 1, 4), (4096, 4, 4, 1))
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [conv2d_3, x_8], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_8.run(buf18, primals_13, 16384, grid=grid(16384), stream=stream0)
        del primals_13
        buf19 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf20 = reinterpret_tensor(buf19, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf19  # reuse
        buf21 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf20, primals_15, primals_14, buf21, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf18, buf21, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 1024, 1, 4), (4096, 4, 4, 1))
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [conv2d_4, x_10], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_8.run(buf23, primals_16, 16384, grid=grid(16384), stream=stream0)
        del primals_16
        buf24 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_10.run(buf25, primals_18, primals_17, buf26, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf23, buf26, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 1, 1, 4), (4, 4, 4, 1))
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_11.run(buf28, primals_19, 16, grid=grid(16), stream=stream0)
        del primals_19
    return (reinterpret_tensor(buf28, (4, 4), (4, 1), 0), buf3, buf8, buf13, buf18, buf23, buf28, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, reinterpret_tensor(primals_1, (4, 1, 1, 4), (4, 4, 4, 1), 0), buf0, buf1, buf3, buf5, buf6, buf8, buf10, buf11, buf13, buf15, buf16, buf18, buf20, buf21, buf23, buf25, buf26, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 4), (4, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
