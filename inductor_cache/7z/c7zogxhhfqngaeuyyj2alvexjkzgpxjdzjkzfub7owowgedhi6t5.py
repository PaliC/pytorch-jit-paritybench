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


# kernel path: inductor_cache/yr/cyrbot5y5wjy6t233ykldr6h2kkrmyvwheghloiv7wnuwdo6o3rk.py
# Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => div, mul, pow_1, pow_2, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_2, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %pow_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %div), kwargs = {})
triton_per_fused__weight_norm_interface_0 = async_compile.triton('triton_per_fused__weight_norm_interface_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 28
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 28*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 28*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lt/cltoucqgui7wjzjfiscxphuklr5y4nrgx33kuparhjtjs4mlzcnc.py
# Topologically Sorted Source Nodes: [input_1, add, reciprocal, mul, sin, pow_1, mul_1, x_1], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add => add
#   input_1 => convolution
#   mul => mul_1
#   mul_1 => mul_2
#   pow_1 => pow_3
#   reciprocal => reciprocal
#   sin => sin
#   x_1 => add_1
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_4, %mul, %primals_3, [1], [3], [1], False, [0], 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_5, 1e-09), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %view), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin, 2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, %pow_3), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view, %mul_2), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_1 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-09
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp6 / tmp5
    tmp8 = tmp3 * tmp2
    tmp9 = tl_math.sin(tmp8)
    tmp10 = tmp9 * tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp2 + tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kd/ckd3lcsp62cc4sobddhav7yo6jqwlyhb6mdgpvv2vqigyhagj232.py
# Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_1 => div_1, mul_3, pow_4, pow_5, sum_2
# Graph fragment:
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_7, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [1, 2], True), kwargs = {})
#   %pow_5 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_6, %pow_5), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_7, %div_1), kwargs = {})
triton_per_fused__weight_norm_interface_2 = async_compile.triton('triton_per_fused__weight_norm_interface_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 16*x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5w/c5wxkujzumt3jbiwpkecw3sis4hia23vu3rpppo6yrvlhq6x6ww5.py
# Topologically Sorted Source Nodes: [input_2, add_2, reciprocal_1, mul_2, sin_1, pow_2, mul_3, x_4], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_2 => add_2
#   input_2 => convolution_1
#   mul_2 => mul_4
#   mul_3 => mul_5
#   pow_2 => pow_6
#   reciprocal_1 => reciprocal_1
#   sin_1 => sin_1
#   x_4 => add_3
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1, %mul_3, %primals_8, [4], [2], [1], True, [0], 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_9, 1e-09), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_2,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, %view_2), kwargs = {})
#   %sin_1 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_4,), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_1, 2), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, %pow_6), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %mul_5), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-09
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp6 / tmp5
    tmp8 = tmp3 * tmp2
    tmp9 = tl_math.sin(tmp8)
    tmp10 = tmp9 * tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp2 + tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fr/cfrcgmalprixaepuq74o3vrhe7y4l3jaefdagutxumw2jbrqguqz.py
# Topologically Sorted Source Nodes: [_weight_norm_2], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_2 => div_2, mul_6, pow_7, pow_8, sum_3
# Graph fragment:
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_11, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   %pow_8 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_10, %pow_8), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_11, %div_2), kwargs = {})
triton_per_fused__weight_norm_interface_4 = async_compile.triton('triton_per_fused__weight_norm_interface_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 14
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 14*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 14*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qf/cqfdtuptvpdrz7tc3y57o7fm4rkxoagfzkp3rloy3hwlqubublza.py
# Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_3 => pow_10, pow_11, sum_4
# Graph fragment:
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_15, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_10, [1, 2], True), kwargs = {})
#   %pow_11 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
triton_poi_fused__weight_norm_interface_5 = async_compile.triton('triton_poi_fused__weight_norm_interface_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2i/c2i2ocq43igqpj5jdbvym7wctrotrve4ovb7qffsmcelbpmxewu6.py
# Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_3 => div_3, mul_9
# Graph fragment:
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_14, %pow_11), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_15, %div_3), kwargs = {})
triton_poi_fused__weight_norm_interface_6 = async_compile.triton('triton_poi_fused__weight_norm_interface_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bm/cbmks6c5wrl7tl3maus42jygu2qd2hc7irgxhtwh673rqgxs5lz4.py
# Topologically Sorted Source Nodes: [input_4, input_5, x_9, add_7, reciprocal_3, mul_6, sin_3, pow_4, mul_7, x_10], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_7 => add_7
#   input_4 => convolution_3
#   input_5 => add_6
#   mul_6 => mul_10
#   mul_7 => mul_11
#   pow_4 => pow_12
#   reciprocal_3 => reciprocal_3
#   sin_3 => sin_3
#   x_10 => add_8
#   x_9 => view_6
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %mul_9, %primals_16, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %view_6 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [4, 2, -1]), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_17, 1e-09), kwargs = {})
#   %reciprocal_3 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_7,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_17, %view_6), kwargs = {})
#   %sin_3 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_10,), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_3, 2), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, %pow_12), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %mul_11), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_7 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp6 = 1e-09
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = tmp5 * tmp4
    tmp11 = tl_math.sin(tmp10)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = tmp4 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ti/ctixd2tuww5p4gv4jlcy5l4qdqzhtldzmwxkher4smbuwar23kzi.py
# Topologically Sorted Source Nodes: [input_5, input_7, input_8, x_15, add_12, reciprocal_5, mul_10, sin_5, pow_6, mul_11, x_16], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_12 => add_12
#   input_5 => add_6
#   input_7 => convolution_5
#   input_8 => add_11
#   mul_10 => mul_16
#   mul_11 => mul_17
#   pow_6 => pow_18
#   reciprocal_5 => reciprocal_5
#   sin_5 => sin_5
#   x_15 => view_10
#   x_16 => add_13
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_10, %mul_15, %primals_24, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %convolution_5), kwargs = {})
#   %view_10 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_11, [4, 2, -1]), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_25, 1e-09), kwargs = {})
#   %reciprocal_5 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_12,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_25, %view_10), kwargs = {})
#   %sin_5 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_16,), kwargs = {})
#   %pow_18 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_5, 2), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, %pow_18), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %mul_17), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_8 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = 1e-09
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp10 / tmp9
    tmp12 = tmp7 * tmp6
    tmp13 = tl_math.sin(tmp12)
    tmp14 = tmp13 * tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp6 + tmp15
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5u/c5ubwinbdbmadbf7ky7pca76gdnw4hgsuuleckyzzo2sq3gmcfmf.py
# Topologically Sorted Source Nodes: [input_5, input_7, input_8, input_10, input_11, x_21, add_17, reciprocal_7, mul_14, sin_7, pow_8, mul_15, x_22], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_17 => add_17
#   input_10 => convolution_7
#   input_11 => add_16
#   input_5 => add_6
#   input_7 => convolution_5
#   input_8 => add_11
#   mul_14 => mul_22
#   mul_15 => mul_23
#   pow_8 => pow_24
#   reciprocal_7 => reciprocal_7
#   sin_7 => sin_7
#   x_21 => view_14
#   x_22 => add_18
# Graph fragment:
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_10, %mul_15, %primals_24, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %convolution_5), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_15, %mul_21, %primals_32, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %convolution_7), kwargs = {})
#   %view_14 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_16, [4, 2, -1]), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_33, 1e-09), kwargs = {})
#   %reciprocal_7 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_17,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_33, %view_14), kwargs = {})
#   %sin_7 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_22,), kwargs = {})
#   %pow_24 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_7, 2), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, %pow_24), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %mul_23), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_9 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x3), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = 1e-09
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = tmp11 * tmp10
    tmp17 = tl_math.sin(tmp16)
    tmp18 = tmp17 * tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = tmp10 + tmp19
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aq/caq47ghf44lps5ymjwnyfzbzr4qbvq5i6pfdv33mkuc2fjrlasaf.py
# Topologically Sorted Source Nodes: [_weight_norm_8], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_8 => div_8, mul_24, pow_25, pow_26, sum_9
# Graph fragment:
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_35, 2), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_25, [1, 2], True), kwargs = {})
#   %pow_26 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_9, 0.5), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_34, %pow_26), kwargs = {})
#   %mul_24 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_35, %div_8), kwargs = {})
triton_per_fused__weight_norm_interface_10 = async_compile.triton('triton_per_fused__weight_norm_interface_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_10(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8*x0), xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 8*x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l2/cl2tfawbbxza634mchf6lnxjh7s52z7m56ngvqkvcnadabeecwmv.py
# Topologically Sorted Source Nodes: [input_12, add_19, reciprocal_8, mul_16, sin_8, pow_9, mul_17, x_25], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_19 => add_19
#   input_12 => convolution_8
#   mul_16 => mul_25
#   mul_17 => mul_26
#   pow_9 => pow_27
#   reciprocal_8 => reciprocal_8
#   sin_8 => sin_8
#   x_25 => add_20
# Graph fragment:
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %mul_24, %primals_36, [4], [2], [1], True, [0], 1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_37, 1e-09), kwargs = {})
#   %reciprocal_8 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_19,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_37, %view_16), kwargs = {})
#   %sin_8 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_25,), kwargs = {})
#   %pow_27 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_8, 2), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, %pow_27), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_16, %mul_26), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = 1e-09
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = tmp5 * tmp3
    tmp11 = tl_math.sin(tmp10)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = tmp3 + tmp13
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nv/cnvsdowl4or5kk3owet6vydo6ltirkp25rutjt25dah27tef24pk.py
# Topologically Sorted Source Nodes: [_weight_norm_9], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_9 => pow_28, pow_29, sum_10
# Graph fragment:
#   %pow_28 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_39, 2), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_28, [1, 2], True), kwargs = {})
#   %pow_29 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_10, 0.5), kwargs = {})
triton_poi_fused__weight_norm_interface_12 = async_compile.triton('triton_poi_fused__weight_norm_interface_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (2))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (3))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (4))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp19 = tl.load(in_ptr0 + (5))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp23 = tl.load(in_ptr0 + (6))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp2 = tmp1 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp8 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp12 * tmp12
    tmp14 = tmp10 + tmp13
    tmp17 = tmp16 * tmp16
    tmp18 = tmp14 + tmp17
    tmp21 = tmp20 * tmp20
    tmp22 = tmp18 + tmp21
    tmp25 = tmp24 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = libdevice.sqrt(tmp26)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/wk/cwktuivemx5y3o53nmwqjipdl7gffoax5dkopgmdlrhb6pewhfmm.py
# Topologically Sorted Source Nodes: [_weight_norm_9], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_9 => div_9, mul_27
# Graph fragment:
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_38, %pow_29), kwargs = {})
#   %mul_27 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_39, %div_9), kwargs = {})
triton_poi_fused__weight_norm_interface_13 = async_compile.triton('triton_poi_fused__weight_norm_interface_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp5 = tmp2 / tmp4
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pd/cpdxltc6wl36ynpbjlcnv7vwcfwhmo54e2v4dvrpnkrh36luy4ru.py
# Topologically Sorted Source Nodes: [_weight_norm_10], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_10 => div_10, mul_30, pow_31, pow_32, sum_11
# Graph fragment:
#   %pow_31 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_43, 2), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_31, [1, 2], True), kwargs = {})
#   %pow_32 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_11, 0.5), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_42, %pow_32), kwargs = {})
#   %mul_30 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_43, %div_10), kwargs = {})
triton_poi_fused__weight_norm_interface_14 = async_compile.triton('triton_poi_fused__weight_norm_interface_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_14(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp2 = tmp1 * tmp1
    tmp3 = libdevice.sqrt(tmp2)
    tmp6 = tmp5 / tmp3
    tmp7 = tmp1 * tmp6
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/bu/cbuti4bqtuum54uuv5a6dtdugvcmme2vlaqfbwwtxx2m6qmy5xtp.py
# Topologically Sorted Source Nodes: [input_14, input_15, x_30, add_24, reciprocal_10, mul_20, sin_10, pow_11, mul_21, x_31], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_24 => add_24
#   input_14 => convolution_10
#   input_15 => add_23
#   mul_20 => mul_31
#   mul_21 => mul_32
#   pow_11 => pow_33
#   reciprocal_10 => reciprocal_10
#   sin_10 => sin_10
#   x_30 => view_20
#   x_31 => add_25
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_22, %mul_30, %primals_44, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %convolution_10), kwargs = {})
#   %view_20 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_23, [4, 1, -1]), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_45, 1e-09), kwargs = {})
#   %reciprocal_10 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_24,), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_45, %view_20), kwargs = {})
#   %sin_10 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_31,), kwargs = {})
#   %pow_33 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_10, 2), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, %pow_33), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_20, %mul_32), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_15 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4 + tmp3
    tmp8 = 1e-09
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp10 / tmp9
    tmp12 = tmp7 * tmp5
    tmp13 = tl_math.sin(tmp12)
    tmp14 = tmp13 * tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp5 + tmp15
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jc/cjczbvrvv7qm55vbjn3n4h2qf5jfyi7xemidsfn5kwsf6tbzpibl.py
# Topologically Sorted Source Nodes: [input_15, input_17, input_18, x_36, add_29, reciprocal_12, mul_24, sin_12, pow_13, mul_25, x_37], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_29 => add_29
#   input_15 => add_23
#   input_17 => convolution_12
#   input_18 => add_28
#   mul_24 => mul_37
#   mul_25 => mul_38
#   pow_13 => pow_39
#   reciprocal_12 => reciprocal_12
#   sin_12 => sin_12
#   x_36 => view_24
#   x_37 => add_30
# Graph fragment:
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %convolution_10), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_27, %mul_36, %primals_52, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %convolution_12), kwargs = {})
#   %view_24 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_28, [4, 1, -1]), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_53, 1e-09), kwargs = {})
#   %reciprocal_12 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_29,), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_53, %view_24), kwargs = {})
#   %sin_12 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_37,), kwargs = {})
#   %pow_39 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_12, 2), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, %pow_39), kwargs = {})
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_24, %mul_38), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_16 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr4 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp6 = tmp3 + tmp5
    tmp7 = tmp2 + tmp6
    tmp10 = 1e-09
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = tmp9 * tmp7
    tmp15 = tl_math.sin(tmp14)
    tmp16 = tmp15 * tmp15
    tmp17 = tmp13 * tmp16
    tmp18 = tmp7 + tmp17
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yi/cyi74oa2j5oji5e2scx7wzeeb7lycwe23l4lt42zorfy2e6zqe5r.py
# Topologically Sorted Source Nodes: [input_15, input_17, input_18, input_20, input_21, x_42, add_34, reciprocal_14, mul_28, sin_14, pow_15, mul_29, x_43], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
# Source node to ATen node mapping:
#   add_34 => add_34
#   input_15 => add_23
#   input_17 => convolution_12
#   input_18 => add_28
#   input_20 => convolution_14
#   input_21 => add_33
#   mul_28 => mul_43
#   mul_29 => mul_44
#   pow_15 => pow_45
#   reciprocal_14 => reciprocal_14
#   sin_14 => sin_14
#   x_42 => view_28
#   x_43 => add_35
# Graph fragment:
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %convolution_10), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_27, %mul_36, %primals_52, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %convolution_12), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_32, %mul_42, %primals_60, [1], [0], [1], False, [0], 1), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %convolution_14), kwargs = {})
#   %view_28 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_33, [4, 1, -1]), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_61, 1e-09), kwargs = {})
#   %reciprocal_14 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_34,), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_61, %view_28), kwargs = {})
#   %sin_14 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_43,), kwargs = {})
#   %pow_45 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sin_14, 2), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_14, %pow_45), kwargs = {})
#   %add_35 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_28, %mul_44), kwargs = {})
triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_17 = async_compile.triton('triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    tmp9 = tl.load(in_ptr4 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp13 = tl.load(in_ptr5 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp6 = tmp3 + tmp5
    tmp7 = tmp2 + tmp6
    tmp11 = tmp8 + tmp10
    tmp12 = tmp7 + tmp11
    tmp15 = 1e-09
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = tmp14 * tmp12
    tmp20 = tl_math.sin(tmp19)
    tmp21 = tmp20 * tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp12 + tmp22
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mj/cmjpdttjgij5h4a6jkmo2a6g4ezo2j7h6j2uoapjkpotst4gef6f.py
# Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.convolution, aten.tanh]
# Source node to ATen node mapping:
#   input_22 => convolution_15
#   input_23 => tanh
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_35, %mul_45, %primals_64, [1], [3], [1], False, [0], 1), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_15,), kwargs = {})
triton_poi_fused_convolution_tanh_18 = async_compile.triton('triton_poi_fused_convolution_tanh_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_tanh_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_tanh_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_2, (4, 4, 7), (28, 7, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_5, (1, 4, 1), (4, 1, 1))
    assert_size_stride(primals_6, (4, 1, 1), (1, 1, 1))
    assert_size_stride(primals_7, (4, 2, 8), (16, 8, 1))
    assert_size_stride(primals_8, (2, ), (1, ))
    assert_size_stride(primals_9, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_10, (2, 1, 1), (1, 1, 1))
    assert_size_stride(primals_11, (2, 2, 7), (14, 7, 1))
    assert_size_stride(primals_12, (2, ), (1, ))
    assert_size_stride(primals_13, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_14, (2, 1, 1), (1, 1, 1))
    assert_size_stride(primals_15, (2, 2, 1), (2, 1, 1))
    assert_size_stride(primals_16, (2, ), (1, ))
    assert_size_stride(primals_17, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_18, (2, 1, 1), (1, 1, 1))
    assert_size_stride(primals_19, (2, 2, 7), (14, 7, 1))
    assert_size_stride(primals_20, (2, ), (1, ))
    assert_size_stride(primals_21, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_22, (2, 1, 1), (1, 1, 1))
    assert_size_stride(primals_23, (2, 2, 1), (2, 1, 1))
    assert_size_stride(primals_24, (2, ), (1, ))
    assert_size_stride(primals_25, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_26, (2, 1, 1), (1, 1, 1))
    assert_size_stride(primals_27, (2, 2, 7), (14, 7, 1))
    assert_size_stride(primals_28, (2, ), (1, ))
    assert_size_stride(primals_29, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_30, (2, 1, 1), (1, 1, 1))
    assert_size_stride(primals_31, (2, 2, 1), (2, 1, 1))
    assert_size_stride(primals_32, (2, ), (1, ))
    assert_size_stride(primals_33, (1, 2, 1), (2, 1, 1))
    assert_size_stride(primals_34, (2, 1, 1), (1, 1, 1))
    assert_size_stride(primals_35, (2, 1, 8), (8, 8, 1))
    assert_size_stride(primals_36, (1, ), (1, ))
    assert_size_stride(primals_37, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_38, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_39, (1, 1, 7), (7, 7, 1))
    assert_size_stride(primals_40, (1, ), (1, ))
    assert_size_stride(primals_41, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_42, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_43, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_44, (1, ), (1, ))
    assert_size_stride(primals_45, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_46, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_47, (1, 1, 7), (7, 7, 1))
    assert_size_stride(primals_48, (1, ), (1, ))
    assert_size_stride(primals_49, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_50, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_51, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_52, (1, ), (1, ))
    assert_size_stride(primals_53, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_54, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_55, (1, 1, 7), (7, 7, 1))
    assert_size_stride(primals_56, (1, ), (1, ))
    assert_size_stride(primals_57, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_58, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_59, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_60, (1, ), (1, ))
    assert_size_stride(primals_61, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_62, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_63, (1, 1, 7), (7, 7, 1))
    assert_size_stride(primals_64, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 1, 1), (1, 1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((4, 4, 7), (28, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf1, primals_2, primals_1, buf2, 4, 28, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_4, buf2, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 4), (16, 4, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, add, reciprocal, mul, sin, pow_1, mul_1, x_1], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_1.run(buf4, primals_3, primals_5, buf5, 64, grid=grid(64), stream=stream0)
        del primals_3
        buf6 = empty_strided_cuda((4, 1, 1), (1, 4, 4), torch.float32)
        buf7 = reinterpret_tensor(buf6, (4, 1, 1), (1, 1, 1), 0); del buf6  # reuse
        buf8 = empty_strided_cuda((4, 2, 8), (16, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_1], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_2.run(buf7, primals_7, primals_6, buf8, 4, 16, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf5, buf8, stride=(4,), padding=(2,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf9, (4, 2, 16), (32, 16, 1))
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, add_2, reciprocal_1, mul_2, sin_1, pow_2, mul_3, x_4], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3.run(buf10, primals_8, primals_9, buf11, 128, grid=grid(128), stream=stream0)
        del primals_8
        buf12 = empty_strided_cuda((2, 1, 1), (1, 2, 2), torch.float32)
        buf13 = reinterpret_tensor(buf12, (2, 1, 1), (1, 1, 1), 0); del buf12  # reuse
        buf14 = empty_strided_cuda((2, 2, 7), (14, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_2], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf13, primals_11, primals_10, buf14, 2, 14, grid=grid(2), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf11, buf14, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf15, (4, 2, 16), (32, 16, 1))
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, add_4, reciprocal_2, mul_4, sin_2, pow_3, mul_5, x_7], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3.run(buf16, primals_12, primals_13, buf17, 128, grid=grid(128), stream=stream0)
        del primals_12
        buf18 = empty_strided_cuda((2, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_5.run(primals_15, buf18, 2, grid=grid(2), stream=stream0)
        buf19 = empty_strided_cuda((2, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_3], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_15, primals_14, buf18, buf19, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf17, buf19, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf20, (4, 2, 16), (32, 16, 1))
        buf21 = buf20; del buf20  # reuse
        buf22 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, x_9, add_7, reciprocal_3, mul_6, sin_3, pow_4, mul_7, x_10], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_7.run(buf21, primals_16, buf10, primals_17, buf22, 128, grid=grid(128), stream=stream0)
        del primals_16
        buf23 = empty_strided_cuda((2, 1, 1), (1, 2, 2), torch.float32)
        buf24 = reinterpret_tensor(buf23, (2, 1, 1), (1, 1, 1), 0); del buf23  # reuse
        buf25 = empty_strided_cuda((2, 2, 7), (14, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_4], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf24, primals_19, primals_18, buf25, 2, 14, grid=grid(2), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf22, buf25, stride=(1,), padding=(9,), dilation=(3,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf26, (4, 2, 16), (32, 16, 1))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, add_9, reciprocal_4, mul_8, sin_4, pow_5, mul_9, x_13], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3.run(buf27, primals_20, primals_21, buf28, 128, grid=grid(128), stream=stream0)
        del primals_20
        buf29 = empty_strided_cuda((2, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_5.run(primals_23, buf29, 2, grid=grid(2), stream=stream0)
        buf30 = empty_strided_cuda((2, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_23, primals_22, buf29, buf30, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf28, buf30, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf31, (4, 2, 16), (32, 16, 1))
        buf32 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        buf33 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_7, input_8, x_15, add_12, reciprocal_5, mul_10, sin_5, pow_6, mul_11, x_16], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_8.run(buf10, buf21, buf31, primals_24, primals_25, buf32, buf33, 128, grid=grid(128), stream=stream0)
        buf34 = empty_strided_cuda((2, 1, 1), (1, 2, 2), torch.float32)
        buf35 = reinterpret_tensor(buf34, (2, 1, 1), (1, 1, 1), 0); del buf34  # reuse
        buf36 = empty_strided_cuda((2, 2, 7), (14, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_6], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf35, primals_27, primals_26, buf36, 2, 14, grid=grid(2), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf33, buf36, stride=(1,), padding=(27,), dilation=(9,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf37, (4, 2, 16), (32, 16, 1))
        buf38 = buf37; del buf37  # reuse
        buf39 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, add_14, reciprocal_6, mul_12, sin_6, pow_7, mul_13, x_19], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_3.run(buf38, primals_28, primals_29, buf39, 128, grid=grid(128), stream=stream0)
        del primals_28
        buf40 = empty_strided_cuda((2, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_7], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_5.run(primals_31, buf40, 2, grid=grid(2), stream=stream0)
        buf41 = empty_strided_cuda((2, 2, 1), (2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_7], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_31, primals_30, buf40, buf41, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf39, buf41, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf42, (4, 2, 16), (32, 16, 1))
        buf43 = buf31; del buf31  # reuse
        buf44 = empty_strided_cuda((4, 2, 16), (32, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_7, input_8, input_10, input_11, x_21, add_17, reciprocal_7, mul_14, sin_7, pow_8, mul_15, x_22], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_9.run(buf43, buf10, buf21, primals_24, buf42, primals_32, primals_33, buf44, 128, grid=grid(128), stream=stream0)
        del buf42
        del primals_24
        del primals_32
        buf45 = empty_strided_cuda((2, 1, 1), (1, 2, 2), torch.float32)
        buf46 = reinterpret_tensor(buf45, (2, 1, 1), (1, 1, 1), 0); del buf45  # reuse
        buf47 = empty_strided_cuda((2, 1, 8), (8, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_8], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_10.run(buf46, primals_35, primals_34, buf47, 2, 8, grid=grid(2), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf44, buf47, stride=(4,), padding=(2,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf48, (4, 1, 64), (64, 64, 1))
        buf49 = buf48; del buf48  # reuse
        buf50 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12, add_19, reciprocal_8, mul_16, sin_8, pow_9, mul_17, x_25], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11.run(buf49, primals_36, primals_37, buf50, 256, grid=grid(256), stream=stream0)
        del primals_36
        buf51 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_9], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_39, buf51, 1, grid=grid(1), stream=stream0)
        buf52 = empty_strided_cuda((1, 1, 7), (7, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_9], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_39, primals_38, buf51, buf52, 7, grid=grid(7), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf50, buf52, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf53, (4, 1, 64), (64, 64, 1))
        buf54 = buf53; del buf53  # reuse
        buf55 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, add_21, reciprocal_9, mul_18, sin_9, pow_10, mul_19, x_28], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11.run(buf54, primals_40, primals_41, buf55, 256, grid=grid(256), stream=stream0)
        del primals_40
        buf56 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf57 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_10], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_14.run(primals_43, primals_42, buf56, buf57, 1, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf55, buf57, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf58, (4, 1, 64), (64, 64, 1))
        buf59 = buf58; del buf58  # reuse
        buf60 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15, x_30, add_24, reciprocal_10, mul_20, sin_10, pow_11, mul_21, x_31], Original ATen: [aten.convolution, aten.add, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_15.run(buf59, primals_44, buf49, primals_45, buf60, 256, grid=grid(256), stream=stream0)
        del primals_44
        buf61 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_11], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_47, buf61, 1, grid=grid(1), stream=stream0)
        buf62 = empty_strided_cuda((1, 1, 7), (7, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_11], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_47, primals_46, buf61, buf62, 7, grid=grid(7), stream=stream0)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf60, buf62, stride=(1,), padding=(9,), dilation=(3,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf63, (4, 1, 64), (64, 64, 1))
        buf64 = buf63; del buf63  # reuse
        buf65 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, add_26, reciprocal_11, mul_22, sin_11, pow_12, mul_23, x_34], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11.run(buf64, primals_48, primals_49, buf65, 256, grid=grid(256), stream=stream0)
        del primals_48
        buf66 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf67 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_12], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_14.run(primals_51, primals_50, buf66, buf67, 1, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf65, buf67, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf68, (4, 1, 64), (64, 64, 1))
        buf69 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        buf70 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_17, input_18, x_36, add_29, reciprocal_12, mul_24, sin_12, pow_13, mul_25, x_37], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_16.run(buf49, buf59, buf68, primals_52, primals_53, buf69, buf70, 256, grid=grid(256), stream=stream0)
        buf71 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_13], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_55, buf71, 1, grid=grid(1), stream=stream0)
        buf72 = empty_strided_cuda((1, 1, 7), (7, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_13], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_55, primals_54, buf71, buf72, 7, grid=grid(7), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf70, buf72, stride=(1,), padding=(27,), dilation=(9,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf73, (4, 1, 64), (64, 64, 1))
        buf74 = buf73; del buf73  # reuse
        buf75 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, add_31, reciprocal_13, mul_26, sin_13, pow_14, mul_27, x_40], Original ATen: [aten.convolution, aten.add, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_11.run(buf74, primals_56, primals_57, buf75, 256, grid=grid(256), stream=stream0)
        del primals_56
        buf76 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf77 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_14], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_14.run(primals_59, primals_58, buf76, buf77, 1, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf75, buf77, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf78, (4, 1, 64), (64, 64, 1))
        buf79 = buf68; del buf68  # reuse
        buf80 = empty_strided_cuda((4, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_17, input_18, input_20, input_21, x_42, add_34, reciprocal_14, mul_28, sin_14, pow_15, mul_29, x_43], Original ATen: [aten.add, aten.convolution, aten.view, aten.reciprocal, aten.mul, aten.sin, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_pow_reciprocal_sin_view_17.run(buf79, buf49, buf59, primals_52, buf78, primals_60, primals_61, buf80, 256, grid=grid(256), stream=stream0)
        del buf78
        del primals_52
        del primals_60
        buf81 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_15], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_63, buf81, 1, grid=grid(1), stream=stream0)
        buf82 = empty_strided_cuda((1, 1, 7), (7, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_15], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_63, primals_62, buf81, buf82, 7, grid=grid(7), stream=stream0)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf80, buf82, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf83, (4, 1, 64), (64, 64, 1))
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.convolution, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_tanh_18.run(buf84, primals_64, 256, grid=grid(256), stream=stream0)
        del primals_64
    return (buf84, buf2, buf8, buf14, buf19, buf25, buf30, buf36, buf41, buf47, buf52, buf57, buf62, buf67, buf72, buf77, buf82, primals_1, primals_2, primals_4, primals_5, primals_6, primals_7, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_25, primals_26, primals_27, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf18, buf19, buf21, buf22, buf24, buf25, buf27, buf28, buf29, buf30, buf32, buf33, buf35, buf36, buf38, buf39, buf40, buf41, buf43, buf44, buf46, buf47, buf49, buf50, buf51, buf52, buf54, buf55, buf56, buf57, buf59, buf60, buf61, buf62, buf64, buf65, buf66, buf67, buf69, buf70, buf71, buf72, buf74, buf75, buf76, buf77, buf79, buf80, buf81, buf82, buf84, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 7), (28, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 2, 8), (16, 8, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2, 2, 7), (14, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2, 2, 7), (14, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((2, 2, 7), (14, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, 2, 1), (2, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((2, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2, 1, 8), (8, 8, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 7), (7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1, 1, 7), (7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1, 1, 7), (7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1, 1, 7), (7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
