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


# kernel path: inductor_cache/pz/cpzgmqhoaoalrph6ppk6lpaojth2plysw4icw6ectjt3umkaf4b5.py
# Topologically Sorted Source Nodes: [_weight_norm, _weight_norm_7], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm => div, mul, pow_1, pow_2, sum_1
#   _weight_norm_7 => mul_13
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_2, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %pow_2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %div), kwargs = {})
#   %mul_13 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %div), kwargs = {})
triton_per_fused__weight_norm_interface_0 = async_compile.triton('triton_per_fused__weight_norm_interface_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 15*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 15*x0), tmp9, rmask & xmask)
    tl.store(out_ptr1 + (r1 + 15*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vm/cvmqv4y3cniclekmgcab6tmjveitnsq5yxkgzsdcfdhdrcjung5g.py
# Topologically Sorted Source Nodes: [_weight_norm_1, _weight_norm_8], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_1 => div_1, mul_2, pow_3, pow_4, sum_2
#   _weight_norm_8 => mul_15
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_6, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1, 2], True), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_5, %pow_4), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %div_1), kwargs = {})
#   %mul_15 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %div_1), kwargs = {})
triton_per_fused__weight_norm_interface_1 = async_compile.triton('triton_per_fused__weight_norm_interface_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 164
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 164*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 164*x0), tmp9, rmask & xmask)
    tl.store(out_ptr1 + (r1 + 164*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a6/ca6bjcsodvanublzu3glyxrxpjyx7ofxyce4scnmhspcxxr2xdir.py
# Topologically Sorted Source Nodes: [_weight_norm_2, _weight_norm_9], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_2 => div_2, mul_4, pow_5, pow_6, sum_3
#   _weight_norm_9 => mul_17
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_9, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1, 2], True), kwargs = {})
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_8, %pow_6), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, %div_2), kwargs = {})
#   %mul_17 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, %div_2), kwargs = {})
triton_per_fused__weight_norm_interface_2 = async_compile.triton('triton_per_fused__weight_norm_interface_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 164
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 164*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 164*x0), tmp9, rmask & xmask)
    tl.store(out_ptr1 + (r1 + 164*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wx/cwxbf4bhkivtbjfbjuofpuknz24jorxyx43lfmylfk6psc5j6psv.py
# Topologically Sorted Source Nodes: [_weight_norm_3, _weight_norm_10], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_10 => mul_19
#   _weight_norm_3 => div_3, mul_6, pow_7, pow_8, sum_4
# Graph fragment:
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_12, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   %pow_8 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_11, %pow_8), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, %div_3), kwargs = {})
#   %mul_19 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, %div_3), kwargs = {})
triton_per_fused__weight_norm_interface_3 = async_compile.triton('triton_per_fused__weight_norm_interface_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 164
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 164*x0), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (r1 + 164*x0), tmp9, rmask & xmask)
    tl.store(out_ptr1 + (r1 + 164*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jc/cjcdsgg7pcoj32r23qdoj4acrnwr2g7rzmf4jydijhvn725ec66u.py
# Topologically Sorted Source Nodes: [_weight_norm_5, _weight_norm_12], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_12 => mul_23
#   _weight_norm_5 => div_5, mul_10, pow_11, pow_12, sum_6
# Graph fragment:
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_18, 2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2], True), kwargs = {})
#   %pow_12 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_6, 0.5), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_17, %pow_12), kwargs = {})
#   %mul_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, %div_5), kwargs = {})
#   %mul_23 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, %div_5), kwargs = {})
triton_red_fused__weight_norm_interface_4 = async_compile.triton('triton_red_fused__weight_norm_interface_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr1 + (r1 + 5120*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q5/cq57rdvnhdcpji4luy5q25xj4c7qd34qjs3hhdu6tvejx4p4xhal.py
# Topologically Sorted Source Nodes: [_weight_norm_6, _weight_norm_13], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_13 => mul_25
#   _weight_norm_6 => div_6, mul_12, pow_13, pow_14, sum_7
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_21, 2), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [1, 2], True), kwargs = {})
#   %pow_14 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 0.5), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_20, %pow_14), kwargs = {})
#   %mul_12 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_21, %div_6), kwargs = {})
#   %mul_25 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_21, %div_6), kwargs = {})
triton_red_fused__weight_norm_interface_5 = async_compile.triton('triton_red_fused__weight_norm_interface_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/af/caf6eqpyhnlvfcgwx3wu325slkelgi5qbu5pxpzaajp4roitmtzc.py
# Topologically Sorted Source Nodes: [_weight_norm_14], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_14 => pow_29, pow_30, sum_15
# Graph fragment:
#   %pow_29 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_25, 2), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_29, [1, 2, 3], True), kwargs = {})
#   %pow_30 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_15, 0.5), kwargs = {})
triton_poi_fused__weight_norm_interface_6 = async_compile.triton('triton_poi_fused__weight_norm_interface_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/tj/ctjseq7uk4yyzeu53he2gc7xb2woupquxcapvadvj4uthussl6xj.py
# Topologically Sorted Source Nodes: [_weight_norm_15, _weight_norm_21], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_15 => div_15, mul_28, pow_31, pow_32, sum_16
#   _weight_norm_21 => mul_39
# Graph fragment:
#   %pow_31 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_28, 2), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_31, [1, 2, 3], True), kwargs = {})
#   %pow_32 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_27, %pow_32), kwargs = {})
#   %mul_28 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_28, %div_15), kwargs = {})
#   %mul_39 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_28, %div_15), kwargs = {})
triton_per_fused__weight_norm_interface_7 = async_compile.triton('triton_per_fused__weight_norm_interface_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (r1 + 160*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p3/cp36ourtc4ewoeeps4uywxuuxcat5vf45kxq5mww4rkbzqhhj5ux.py
# Topologically Sorted Source Nodes: [_weight_norm_16, _weight_norm_22], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_16 => div_16, mul_30, pow_33, pow_34, sum_17
#   _weight_norm_22 => mul_41
# Graph fragment:
#   %pow_33 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_31, 2), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_33, [1, 2, 3], True), kwargs = {})
#   %pow_34 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_17, 0.5), kwargs = {})
#   %div_16 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_30, %pow_34), kwargs = {})
#   %mul_30 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_31, %div_16), kwargs = {})
#   %mul_41 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_31, %div_16), kwargs = {})
triton_per_fused__weight_norm_interface_8 = async_compile.triton('triton_per_fused__weight_norm_interface_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tl.store(out_ptr1 + (r1 + 640*x0), tmp9, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/wz/cwzcanunqk7vaykjiqwxreyeuwfehihn53kim7lrs3foorz5yvfo.py
# Topologically Sorted Source Nodes: [_weight_norm_17, _weight_norm_23], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_17 => div_17, mul_32, pow_35, pow_36, sum_18
#   _weight_norm_23 => mul_43
# Graph fragment:
#   %pow_35 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_34, 2), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_35, [1, 2, 3], True), kwargs = {})
#   %pow_36 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_18, 0.5), kwargs = {})
#   %div_17 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_33, %pow_36), kwargs = {})
#   %mul_32 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_34, %div_17), kwargs = {})
#   %mul_43 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_34, %div_17), kwargs = {})
triton_red_fused__weight_norm_interface_9 = async_compile.triton('triton_red_fused__weight_norm_interface_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr1 + (r1 + 2560*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gg/cggkvu7zq3disthhepr5busj42iu42wxh3f2uhinmnt43shbhfps.py
# Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_54 => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad1d_10 = async_compile.triton('triton_poi_fused_reflection_pad1d_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = xindex // 66
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j7/cj72z2szw7sx7tl2pnjgbg4t7zt3eef46g5ofelcod3sn23wx7hg.py
# Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_82 => _unsafe_index_2
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_5]), kwargs = {})
triton_poi_fused_reflection_pad1d_11 = async_compile.triton('triton_poi_fused_reflection_pad1d_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 260
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 65)
    x1 = xindex // 65
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b7/cb7v3gc6z7qmaptqg2j2i2fsouog6cmaqxgffb34jktuqlkzmvrr.py
# Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_110 => _unsafe_index_4
# Graph fragment:
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_9]), kwargs = {})
triton_poi_fused_reflection_pad1d_12 = async_compile.triton('triton_poi_fused_reflection_pad1d_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 70)
    x1 = xindex // 70
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ov/covu3y6wz3blvblaouz2delbx47xg6wdlvos3ty5h6yvwwdqvos7.py
# Topologically Sorted Source Nodes: [_weight_norm_14, _weight_norm_20], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   _weight_norm_14 => div_14, mul_26
#   _weight_norm_20 => mul_37
# Graph fragment:
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_24, %pow_30), kwargs = {})
#   %mul_26 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_25, %div_14), kwargs = {})
#   %mul_37 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_25, %div_14), kwargs = {})
triton_poi_fused__weight_norm_interface_13 = async_compile.triton('triton_poi_fused__weight_norm_interface_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tq/ctq3l7wjacjtzcmnluja7d6oyt7hmg56bh27lvwwnvyyixv6dnxc.py
# Topologically Sorted Source Nodes: [x, x_1, x_14, x_15], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => gt, mul_1, where
#   x_14 => convolution_7
#   x_15 => gt_6, mul_14, where_6
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_4, %mul, %primals_3, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_1), kwargs = {})
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_23, %mul_13, %primals_3, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt_6 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_7, 0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_7, 0.1), kwargs = {})
#   %where_6 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %convolution_7, %mul_14), kwargs = {})
triton_poi_fused_convolution_leaky_relu_14 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_14(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/nq/cnq7hqtvuadtwvpfv6mmbfps63ictdjd4lfud7d353mlqvk5hnph.py
# Topologically Sorted Source Nodes: [x_56, x_57, x_70, x_71], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_56 => convolution_26
#   x_57 => gt_22, mul_49, where_22
#   x_70 => convolution_32
#   x_71 => gt_27, mul_60, where_27
# Graph fragment:
#   %convolution_26 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_6, %mul_48, %primals_44, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_22 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_26, 0), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_26, 0.1), kwargs = {})
#   %where_22 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_22, %convolution_26, %mul_49), kwargs = {})
#   %convolution_32 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_8, %mul_59, %primals_44, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_27 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_32, 0), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_32, 0.1), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_27, %convolution_32, %mul_60), kwargs = {})
triton_poi_fused_convolution_leaky_relu_15 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_15(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 24) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5seioidzlatktm6pcbysdrfjfq3xa5d6psyzdoifddl253gcxto.py
# Topologically Sorted Source Nodes: [x_84, x_85, x_98, x_99], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_84 => convolution_38
#   x_85 => gt_32, mul_71, where_32
#   x_98 => convolution_44
#   x_99 => gt_37, mul_82, where_37
# Graph fragment:
#   %convolution_38 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_10, %mul_70, %primals_62, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_32 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_38, 0), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_38, 0.1), kwargs = {})
#   %where_32 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_32, %convolution_38, %mul_71), kwargs = {})
#   %convolution_44 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_12, %mul_81, %primals_62, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_37 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_44, 0), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_44, 0.1), kwargs = {})
#   %where_37 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_37, %convolution_44, %mul_82), kwargs = {})
triton_poi_fused_convolution_leaky_relu_16 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_16(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 25) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z2/cz2uim7qbv5n7pbvggeradtujeg2ppo7er7cl4i5jspngt6djlal.py
# Topologically Sorted Source Nodes: [x_112, x_113, x_126, x_127], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_112 => convolution_50
#   x_113 => gt_42, mul_93, where_42
#   x_126 => convolution_56
#   x_127 => gt_47, mul_104, where_47
# Graph fragment:
#   %convolution_50 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_14, %mul_92, %primals_80, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_42 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_50, 0), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_50, 0.1), kwargs = {})
#   %where_42 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_42, %convolution_50, %mul_93), kwargs = {})
#   %convolution_56 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_16, %mul_103, %primals_80, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_47 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_56, 0), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_56, 0.1), kwargs = {})
#   %where_47 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_47, %convolution_56, %mul_104), kwargs = {})
triton_poi_fused_convolution_leaky_relu_17 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_17(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 28) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e6/ce6tzmci2zz3v3s3uhui75bfk5fzmusdrmbn4ayn4j4zpaewwtdp.py
# Topologically Sorted Source Nodes: [x_140, x_141, x_154, x_155], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_140 => convolution_62
#   x_141 => gt_52, mul_115, where_52
#   x_154 => convolution_68
#   x_155 => gt_57, mul_126, where_57
# Graph fragment:
#   %convolution_62 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_18, %mul_114, %primals_98, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_52 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_62, 0), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_62, 0.1), kwargs = {})
#   %where_52 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_52, %convolution_62, %mul_115), kwargs = {})
#   %convolution_68 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_20, %mul_125, %primals_98, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_57 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_68, 0), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_68, 0.1), kwargs = {})
#   %where_57 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_57, %convolution_68, %mul_126), kwargs = {})
triton_poi_fused_convolution_leaky_relu_18 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_18(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 22) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/43/c43xj4cezrogguv2jrnrlbug2xkdflnatjiblshqrpcfyurzrzws.py
# Topologically Sorted Source Nodes: [x_2, x_3, x_16, x_17], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_16 => convolution_8
#   x_17 => gt_7, mul_16, where_7
#   x_2 => convolution_1
#   x_3 => gt_1, mul_3, where_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %mul_2, %primals_7, [4], [20], [1], False, [0], 4), kwargs = {})
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.1), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %mul_3), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_6, %mul_15, %primals_7, [4], [20], [1], False, [0], 4), kwargs = {})
#   %gt_7 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_8, 0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_8, 0.1), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %convolution_8, %mul_16), kwargs = {})
triton_poi_fused_convolution_leaky_relu_19 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_19(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/so/csoybliw4xegtug6hsfi5uqcbk3mpzucrlbqxvlnragoeicrj6yo.py
# Topologically Sorted Source Nodes: [x_58, x_59, x_72, x_73], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_58 => convolution_27
#   x_59 => gt_23, mul_51, where_23
#   x_72 => convolution_33
#   x_73 => gt_28, mul_62, where_28
# Graph fragment:
#   %convolution_27 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_22, %mul_50, %primals_47, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_23 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_27, 0), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_27, 0.1), kwargs = {})
#   %where_23 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_23, %convolution_27, %mul_51), kwargs = {})
#   %convolution_33 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_27, %mul_61, %primals_47, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_28 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_33, 0), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_33, 0.1), kwargs = {})
#   %where_28 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_28, %convolution_33, %mul_62), kwargs = {})
triton_poi_fused_convolution_leaky_relu_20 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_20(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 9) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tl/ctlhbt5ol2loyhanbttckkzosusbzw3go2dnxh2im4fkwebxoscc.py
# Topologically Sorted Source Nodes: [x_86, x_87, x_100, x_101], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_100 => convolution_45
#   x_101 => gt_38, mul_84, where_38
#   x_86 => convolution_39
#   x_87 => gt_33, mul_73, where_33
# Graph fragment:
#   %convolution_39 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_32, %mul_72, %primals_65, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_33 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_39, 0), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_39, 0.1), kwargs = {})
#   %where_33 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_33, %convolution_39, %mul_73), kwargs = {})
#   %convolution_45 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_37, %mul_83, %primals_65, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_38 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_45, 0), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_45, 0.1), kwargs = {})
#   %where_38 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_38, %convolution_45, %mul_84), kwargs = {})
triton_poi_fused_convolution_leaky_relu_21 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_21(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 10) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3f/c3fogcnlpp7sbkbhzhaerhtdyqwolzdypzfewszxp3bsy343fpll.py
# Topologically Sorted Source Nodes: [x_114, x_115, x_128, x_129], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_114 => convolution_51
#   x_115 => gt_43, mul_95, where_43
#   x_128 => convolution_57
#   x_129 => gt_48, mul_106, where_48
# Graph fragment:
#   %convolution_51 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_42, %mul_94, %primals_83, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_43 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_51, 0), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_51, 0.1), kwargs = {})
#   %where_43 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_43, %convolution_51, %mul_95), kwargs = {})
#   %convolution_57 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_47, %mul_105, %primals_83, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_48 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_57, 0), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_57, 0.1), kwargs = {})
#   %where_48 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_48, %convolution_57, %mul_106), kwargs = {})
triton_poi_fused_convolution_leaky_relu_22 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_22(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 14) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sp/cspyidlq7gwgebjqj5f4yu4vvsbab6aaw6xjpxjev3v4ind57lrk.py
# Topologically Sorted Source Nodes: [x_142, x_143, x_156, x_157], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_142 => convolution_63
#   x_143 => gt_53, mul_117, where_53
#   x_156 => convolution_69
#   x_157 => gt_58, mul_128, where_58
# Graph fragment:
#   %convolution_63 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_52, %mul_116, %primals_101, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_53 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_63, 0), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_63, 0.1), kwargs = {})
#   %where_53 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_53, %convolution_63, %mul_117), kwargs = {})
#   %convolution_69 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_57, %mul_127, %primals_101, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_58 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_69, 0), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_69, 0.1), kwargs = {})
#   %where_58 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_58, %convolution_69, %mul_128), kwargs = {})
triton_poi_fused_convolution_leaky_relu_23 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_23(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 11) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2i/c2iy7kuzxligdgvybbnowa5rgi5do33ovarc2nrztb3fjom5xf6z.py
# Topologically Sorted Source Nodes: [x_4, x_5, x_18, x_19], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_18 => convolution_9
#   x_19 => gt_8, mul_18, where_8
#   x_4 => convolution_2
#   x_5 => gt_2, mul_5, where_2
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_1, %mul_4, %primals_10, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_2 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_5), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_7, %mul_17, %primals_10, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_8 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_9, 0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_9, 0.1), kwargs = {})
#   %where_8 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %convolution_9, %mul_18), kwargs = {})
triton_poi_fused_convolution_leaky_relu_24 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_24(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/yz/cyzfeqvlqavcbjepvdv3q5b4tohhphuorhnthaujvbkuxeyp2yov.py
# Topologically Sorted Source Nodes: [x_31, x_32, x_44, x_45], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_31 => convolution_15
#   x_32 => gt_13, mul_29, where_13
#   x_44 => convolution_21
#   x_45 => gt_18, mul_40, where_18
# Graph fragment:
#   %convolution_15 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_12, %mul_28, %primals_29, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_13 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_15, 0), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 0.1), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %convolution_15, %mul_29), kwargs = {})
#   %convolution_21 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_17, %mul_39, %primals_29, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_18 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_21, 0), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_21, 0.1), kwargs = {})
#   %where_18 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_18, %convolution_21, %mul_40), kwargs = {})
triton_poi_fused_convolution_leaky_relu_25 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_25(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 8) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/4q/c4q2ze4wts6ryrb4opmmg4svoikrtbbdbidytr755ffxe6pmkbup.py
# Topologically Sorted Source Nodes: [x_60, x_61, x_74, x_75], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_60 => convolution_28
#   x_61 => gt_24, mul_53, where_24
#   x_74 => convolution_34
#   x_75 => gt_29, mul_64, where_29
# Graph fragment:
#   %convolution_28 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_23, %mul_52, %primals_50, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_24 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_28, 0), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_28, 0.1), kwargs = {})
#   %where_24 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_24, %convolution_28, %mul_53), kwargs = {})
#   %convolution_34 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_28, %mul_63, %primals_50, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_29 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_34, 0), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_34, 0.1), kwargs = {})
#   %where_29 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_29, %convolution_34, %mul_64), kwargs = {})
triton_poi_fused_convolution_leaky_relu_26 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_26(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3r/c3rmrwq64tycwffcuuxvlwg6xeeghrbman6v6ktv25bqh5qhspbw.py
# Topologically Sorted Source Nodes: [x_88, x_89, x_102, x_103], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_102 => convolution_46
#   x_103 => gt_39, mul_86, where_39
#   x_88 => convolution_40
#   x_89 => gt_34, mul_75, where_34
# Graph fragment:
#   %convolution_40 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_33, %mul_74, %primals_68, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_34 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_40, 0), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_40, 0.1), kwargs = {})
#   %where_34 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_34, %convolution_40, %mul_75), kwargs = {})
#   %convolution_46 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_38, %mul_85, %primals_68, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_39 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_46, 0), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_46, 0.1), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_39, %convolution_46, %mul_86), kwargs = {})
triton_poi_fused_convolution_leaky_relu_27 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_27(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 5) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l4/cl4ynpiy3yhbw25y36xvgwfugk3yimuclczy2t3tib5uyim6fakw.py
# Topologically Sorted Source Nodes: [x_116, x_117, x_130, x_131], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_116 => convolution_52
#   x_117 => gt_44, mul_97, where_44
#   x_130 => convolution_58
#   x_131 => gt_49, mul_108, where_49
# Graph fragment:
#   %convolution_52 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_43, %mul_96, %primals_86, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_44 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_52, 0), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_52, 0.1), kwargs = {})
#   %where_44 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_44, %convolution_52, %mul_97), kwargs = {})
#   %convolution_58 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_48, %mul_107, %primals_86, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_49 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_58, 0), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_58, 0.1), kwargs = {})
#   %where_49 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_49, %convolution_58, %mul_108), kwargs = {})
triton_poi_fused_convolution_leaky_relu_28 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_28', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_28(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 7) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hc/chctf77tlysxdntacg43pbx4wi2l62qjmamtkux7punrdro63wog.py
# Topologically Sorted Source Nodes: [x_144, x_145, x_158, x_159], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_144 => convolution_64
#   x_145 => gt_54, mul_119, where_54
#   x_158 => convolution_70
#   x_159 => gt_59, mul_130, where_59
# Graph fragment:
#   %convolution_64 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_53, %mul_118, %primals_104, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_54 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_64, 0), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_64, 0.1), kwargs = {})
#   %where_54 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_54, %convolution_64, %mul_119), kwargs = {})
#   %convolution_70 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_58, %mul_129, %primals_104, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_59 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_70, 0), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_70, 0.1), kwargs = {})
#   %where_59 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_59, %convolution_70, %mul_130), kwargs = {})
triton_poi_fused_convolution_leaky_relu_29 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_29(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 11) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ex/cexb5d276ac3i37ounh7pkb6tregfrqxkvl7sjei4j6k5bud2rsq.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_20, x_21], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_20 => convolution_10
#   x_21 => gt_9, mul_20, where_9
#   x_6 => convolution_3
#   x_7 => gt_3, mul_7, where_3
# Graph fragment:
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %mul_6, %primals_13, [4], [20], [1], False, [0], 64), kwargs = {})
#   %gt_3 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 0.1), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_3, %mul_7), kwargs = {})
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_8, %mul_19, %primals_13, [4], [20], [1], False, [0], 64), kwargs = {})
#   %gt_9 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_10, 0), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_10, 0.1), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %convolution_10, %mul_20), kwargs = {})
triton_poi_fused_convolution_leaky_relu_30 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_30(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x2), tmp4, None)
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr1 + (x2), tmp10, None)
    tl.store(in_out_ptr1 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/tg/ctg533x4kbe6mt4puvotwv35lvww7bw3vf7zngz6kgarglrvo6hi.py
# Topologically Sorted Source Nodes: [x_33, x_34, x_46, x_47], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_33 => convolution_16
#   x_34 => gt_14, mul_31, where_14
#   x_46 => convolution_22
#   x_47 => gt_19, mul_42, where_19
# Graph fragment:
#   %convolution_16 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_13, %mul_30, %primals_32, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_14 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_16, 0), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_16, 0.1), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %convolution_16, %mul_31), kwargs = {})
#   %convolution_22 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_18, %mul_41, %primals_32, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_19 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_22, 0), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_22, 0.1), kwargs = {})
#   %where_19 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_19, %convolution_22, %mul_42), kwargs = {})
triton_poi_fused_convolution_leaky_relu_31 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_31(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/nc/cncyrqlpherku7u4agiymw5mnkngzcq6plyvyvvcihxbj34gg6gk.py
# Topologically Sorted Source Nodes: [x_62, x_63, x_76, x_77], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_62 => convolution_29
#   x_63 => gt_25, mul_55, where_25
#   x_76 => convolution_35
#   x_77 => gt_30, mul_66, where_30
# Graph fragment:
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_24, %mul_54, %primals_53, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_25 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_29, 0), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_29, 0.1), kwargs = {})
#   %where_25 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %convolution_29, %mul_55), kwargs = {})
#   %convolution_35 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_29, %mul_65, %primals_53, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_30 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_35, 0), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_35, 0.1), kwargs = {})
#   %where_30 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_30, %convolution_35, %mul_66), kwargs = {})
triton_poi_fused_convolution_leaky_relu_32 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_32(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/5w/c5wcby5w5mb6fmc7b2f24c7e2je74yt7idlytvykw54jyiythfmv.py
# Topologically Sorted Source Nodes: [x_90, x_91, x_104, x_105], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_104 => convolution_47
#   x_105 => gt_40, mul_88, where_40
#   x_90 => convolution_41
#   x_91 => gt_35, mul_77, where_35
# Graph fragment:
#   %convolution_41 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_34, %mul_76, %primals_71, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_35 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_41, 0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_41, 0.1), kwargs = {})
#   %where_35 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_35, %convolution_41, %mul_77), kwargs = {})
#   %convolution_47 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_39, %mul_87, %primals_71, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_40 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_47, 0), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_47, 0.1), kwargs = {})
#   %where_40 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_40, %convolution_47, %mul_88), kwargs = {})
triton_poi_fused_convolution_leaky_relu_33 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_33', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_33(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 5) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/b3/cb3janm7ap4ef4hhbp267p5f4jdwo54lg4wai4ku5svhyjnnvelp.py
# Topologically Sorted Source Nodes: [x_118, x_119, x_132, x_133], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_118 => convolution_53
#   x_119 => gt_45, mul_99, where_45
#   x_132 => convolution_59
#   x_133 => gt_50, mul_110, where_50
# Graph fragment:
#   %convolution_53 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_44, %mul_98, %primals_89, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_45 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_53, 0), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_53, 0.1), kwargs = {})
#   %where_45 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_45, %convolution_53, %mul_99), kwargs = {})
#   %convolution_59 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_49, %mul_109, %primals_89, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_50 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_59, 0), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_59, 0.1), kwargs = {})
#   %where_50 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_50, %convolution_59, %mul_110), kwargs = {})
triton_poi_fused_convolution_leaky_relu_34 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_34', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_34(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 7) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/lh/clhpmst2m6jpx3e7ucvmjatsjjnp4vqytk6qtlu2opivwvmfgjr4.py
# Topologically Sorted Source Nodes: [x_146, x_147, x_160, x_161], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_146 => convolution_65
#   x_147 => gt_55, mul_121, where_55
#   x_160 => convolution_71
#   x_161 => gt_60, mul_132, where_60
# Graph fragment:
#   %convolution_65 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_54, %mul_120, %primals_107, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_55 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_65, 0), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_65, 0.1), kwargs = {})
#   %where_55 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_55, %convolution_65, %mul_121), kwargs = {})
#   %convolution_71 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_59, %mul_131, %primals_107, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_60 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_71, 0), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_71, 0.1), kwargs = {})
#   %where_60 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_60, %convolution_71, %mul_132), kwargs = {})
triton_poi_fused_convolution_leaky_relu_35 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_35', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_35(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 11) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/kd/ckdhonrzryuqjjka5wovw5ni5u5i34wssbyqwoy3raqxxbio54j6.py
# Topologically Sorted Source Nodes: [x_35, x_36, x_48, x_49], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_35 => convolution_17
#   x_36 => gt_15, mul_33, where_15
#   x_48 => convolution_23
#   x_49 => gt_20, mul_44, where_20
# Graph fragment:
#   %convolution_17 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_14, %mul_32, %primals_35, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_15 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_17, 0), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_17, 0.1), kwargs = {})
#   %where_15 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_15, %convolution_17, %mul_33), kwargs = {})
#   %convolution_23 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_19, %mul_43, %primals_35, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_20 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_23, 0), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_23, 0.1), kwargs = {})
#   %where_20 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_20, %convolution_23, %mul_44), kwargs = {})
triton_poi_fused_convolution_leaky_relu_36 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_36', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_36(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 2) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnjyxrnl552y46szbtqpqnlxsmhtx6h3iozs7uhbouotabekgiyj.py
# Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_66 => convolution_31
# Graph fragment:
#   %convolution_31 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_26, %mul_58, %primals_59, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_37 = async_compile.triton('triton_poi_fused_convolution_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
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


# kernel path: inductor_cache/cc/ccclgbskwqn4abigo3nwt6amlmoeose376ybswbqlbzqwkbixjv6.py
# Topologically Sorted Source Nodes: [x_94], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_94 => convolution_43
# Graph fragment:
#   %convolution_43 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_36, %mul_80, %primals_77, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
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


# kernel path: inductor_cache/zl/czlpm7b2lqwbgmnayf6wejdyzlm342kdwrb3pnnzwevsabvjifcs.py
# Topologically Sorted Source Nodes: [x_122], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_122 => convolution_55
# Graph fragment:
#   %convolution_55 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_46, %mul_102, %primals_95, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_39 = async_compile.triton('triton_poi_fused_convolution_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_39(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28
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


# kernel path: inductor_cache/bz/cbzvvjouwg4vciqwatfrmph2k7k334vs37bpzbvhazesrnnqmomq.py
# Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_150 => convolution_67
# Graph fragment:
#   %convolution_67 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_56, %mul_124, %primals_113, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_40 = async_compile.triton('triton_poi_fused_convolution_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_40(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 44
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


# kernel path: inductor_cache/v3/cv3qt5cj56nbjqv6vckbh3mozgq7akuwy4qmjnpxdn5uszggldds.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_12 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_5, %mul_12, %primals_22, [1], [1], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_41 = async_compile.triton('triton_poi_fused_convolution_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_41(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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


# kernel path: inductor_cache/dl/cdl4kdvmo6ccd3twod5vj6n2msv2nwpuuepe4uvbvnue5gqqi4kt.py
# Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_39 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_16, %mul_36, %primals_41, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_42 = async_compile.triton('triton_poi_fused_convolution_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_42(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113 = args
    args.clear()
    assert_size_stride(primals_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_2, (16, 1, 15), (15, 15, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_5, (64, 1, 1), (1, 1, 1))
    assert_size_stride(primals_6, (64, 4, 41), (164, 41, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_9, (256, 4, 41), (164, 41, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_12, (1024, 4, 41), (164, 41, 1))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_15, (1024, 4, 41), (164, 41, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    assert_size_stride(primals_17, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_18, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_21, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_22, (1, ), (1, ))
    assert_size_stride(primals_23, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_24, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_25, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_27, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_28, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_31, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_34, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_37, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_38, (1024, ), (1, ))
    assert_size_stride(primals_39, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_40, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_41, (1, ), (1, ))
    assert_size_stride(primals_42, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_43, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_46, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_49, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_52, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_53, (1024, ), (1, ))
    assert_size_stride(primals_54, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_55, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_56, (1024, ), (1, ))
    assert_size_stride(primals_57, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_58, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_59, (1, ), (1, ))
    assert_size_stride(primals_60, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_61, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_62, (32, ), (1, ))
    assert_size_stride(primals_63, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_64, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_67, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_70, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_71, (1024, ), (1, ))
    assert_size_stride(primals_72, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_73, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_74, (1024, ), (1, ))
    assert_size_stride(primals_75, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_76, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_77, (1, ), (1, ))
    assert_size_stride(primals_78, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_79, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_80, (32, ), (1, ))
    assert_size_stride(primals_81, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_82, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_85, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_88, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_89, (1024, ), (1, ))
    assert_size_stride(primals_90, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_91, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_94, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_95, (1, ), (1, ))
    assert_size_stride(primals_96, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_97, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_98, (32, ), (1, ))
    assert_size_stride(primals_99, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_100, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_103, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_106, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_107, (1024, ), (1, ))
    assert_size_stride(primals_108, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_109, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_112, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_113, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1, 1), (1, 16, 16), torch.float32)
        buf1 = reinterpret_tensor(buf0, (16, 1, 1), (1, 1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((16, 1, 15), (15, 15, 1), torch.float32)
        buf41 = empty_strided_cuda((16, 1, 15), (15, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm, _weight_norm_7], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf1, primals_2, primals_1, buf2, buf41, 16, 15, grid=grid(16), stream=stream0)
        buf6 = empty_strided_cuda((64, 1, 1), (1, 64, 64), torch.float32)
        buf7 = reinterpret_tensor(buf6, (64, 1, 1), (1, 1, 1), 0); del buf6  # reuse
        buf8 = empty_strided_cuda((64, 4, 41), (164, 41, 1), torch.float32)
        buf45 = empty_strided_cuda((64, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_1, _weight_norm_8], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_1.run(buf7, primals_6, primals_5, buf8, buf45, 64, 164, grid=grid(64), stream=stream0)
        buf12 = empty_strided_cuda((256, 1, 1), (1, 256, 256), torch.float32)
        buf13 = reinterpret_tensor(buf12, (256, 1, 1), (1, 1, 1), 0); del buf12  # reuse
        buf14 = empty_strided_cuda((256, 4, 41), (164, 41, 1), torch.float32)
        buf49 = empty_strided_cuda((256, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_2, _weight_norm_9], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_2.run(buf13, primals_9, primals_8, buf14, buf49, 256, 164, grid=grid(256), stream=stream0)
        buf18 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf19 = reinterpret_tensor(buf18, (1024, 1, 1), (1, 1, 1), 0); del buf18  # reuse
        buf20 = empty_strided_cuda((1024, 4, 41), (164, 41, 1), torch.float32)
        buf53 = empty_strided_cuda((1024, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_3, _weight_norm_10], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf19, primals_12, primals_11, buf20, buf53, 1024, 164, grid=grid(1024), stream=stream0)
        buf24 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf25 = reinterpret_tensor(buf24, (1024, 1, 1), (1, 1, 1), 0); del buf24  # reuse
        buf26 = empty_strided_cuda((1024, 4, 41), (164, 41, 1), torch.float32)
        buf57 = empty_strided_cuda((1024, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_4, _weight_norm_11], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_3.run(buf25, primals_15, primals_14, buf26, buf57, 1024, 164, grid=grid(1024), stream=stream0)
        buf30 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf31 = reinterpret_tensor(buf30, (1024, 1, 1), (1, 1, 1), 0); del buf30  # reuse
        buf32 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        buf61 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_5, _weight_norm_12], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_4.run(buf31, primals_18, primals_17, buf32, buf61, 1024, 5120, grid=grid(1024), stream=stream0)
        buf36 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf37 = buf36; del buf36  # reuse
        buf38 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        buf65 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_6, _weight_norm_13], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf37, primals_21, primals_20, buf38, buf65, 1, 3072, grid=grid(1), stream=stream0)
        buf68 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_14], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_25, buf68, 32, grid=grid(32), stream=stream0)
        buf73 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf74 = reinterpret_tensor(buf73, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf73  # reuse
        buf75 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        buf106 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_15, _weight_norm_21], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_7.run(buf74, primals_28, primals_27, buf75, buf106, 128, 160, grid=grid(128), stream=stream0)
        buf79 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf80 = reinterpret_tensor(buf79, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf79  # reuse
        buf81 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        buf110 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_16, _weight_norm_22], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_8.run(buf80, primals_31, primals_30, buf81, buf110, 512, 640, grid=grid(512), stream=stream0)
        buf85 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf86 = reinterpret_tensor(buf85, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf85  # reuse
        buf87 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        buf114 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_17, _weight_norm_23], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf86, primals_34, primals_33, buf87, buf114, 1024, 2560, grid=grid(1024), stream=stream0)
        buf91 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf92 = reinterpret_tensor(buf91, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf91  # reuse
        buf93 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        buf118 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_18, _weight_norm_24], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_4.run(buf92, primals_37, primals_36, buf93, buf118, 1024, 5120, grid=grid(1024), stream=stream0)
        buf97 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf98 = buf97; del buf97  # reuse
        buf99 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        buf122 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_19, _weight_norm_25], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf98, primals_40, primals_39, buf99, buf122, 1, 3072, grid=grid(1), stream=stream0)
        buf125 = empty_strided_cuda((4, 1, 66), (66, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_10.run(primals_4, buf125, 264, grid=grid(264), stream=stream0)
        buf126 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_26], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_43, buf126, 32, grid=grid(32), stream=stream0)
        buf160 = empty_strided_cuda((4, 1, 66), (66, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_10.run(primals_23, buf160, 264, grid=grid(264), stream=stream0)
        buf131 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf132 = reinterpret_tensor(buf131, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf131  # reuse
        buf133 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        buf165 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_27, _weight_norm_33], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_7.run(buf132, primals_46, primals_45, buf133, buf165, 128, 160, grid=grid(128), stream=stream0)
        buf137 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf138 = reinterpret_tensor(buf137, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf137  # reuse
        buf139 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        buf169 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_28, _weight_norm_34], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_8.run(buf138, primals_49, primals_48, buf139, buf169, 512, 640, grid=grid(512), stream=stream0)
        buf143 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf144 = reinterpret_tensor(buf143, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf143  # reuse
        buf145 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        buf173 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_29, _weight_norm_35], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf144, primals_52, primals_51, buf145, buf173, 1024, 2560, grid=grid(1024), stream=stream0)
        buf149 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf150 = reinterpret_tensor(buf149, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf149  # reuse
        buf151 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        buf177 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_30, _weight_norm_36], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_4.run(buf150, primals_55, primals_54, buf151, buf177, 1024, 5120, grid=grid(1024), stream=stream0)
        buf155 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf156 = buf155; del buf155  # reuse
        buf157 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        buf181 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_31, _weight_norm_37], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf156, primals_58, primals_57, buf157, buf181, 1, 3072, grid=grid(1), stream=stream0)
        buf184 = empty_strided_cuda((4, 1, 65), (65, 65, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_11.run(primals_4, buf184, 260, grid=grid(260), stream=stream0)
        buf185 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_38], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_61, buf185, 32, grid=grid(32), stream=stream0)
        buf219 = empty_strided_cuda((4, 1, 65), (65, 65, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_11.run(primals_23, buf219, 260, grid=grid(260), stream=stream0)
        buf190 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf191 = reinterpret_tensor(buf190, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf190  # reuse
        buf192 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        buf224 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_39, _weight_norm_45], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_7.run(buf191, primals_64, primals_63, buf192, buf224, 128, 160, grid=grid(128), stream=stream0)
        buf196 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf197 = reinterpret_tensor(buf196, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf196  # reuse
        buf198 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        buf228 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_40, _weight_norm_46], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_8.run(buf197, primals_67, primals_66, buf198, buf228, 512, 640, grid=grid(512), stream=stream0)
        buf202 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf203 = reinterpret_tensor(buf202, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf202  # reuse
        buf204 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        buf232 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_41, _weight_norm_47], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf203, primals_70, primals_69, buf204, buf232, 1024, 2560, grid=grid(1024), stream=stream0)
        buf208 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf209 = reinterpret_tensor(buf208, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf208  # reuse
        buf210 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        buf236 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_42, _weight_norm_48], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_4.run(buf209, primals_73, primals_72, buf210, buf236, 1024, 5120, grid=grid(1024), stream=stream0)
        buf214 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf215 = buf214; del buf214  # reuse
        buf216 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        buf240 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_43, _weight_norm_49], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf215, primals_76, primals_75, buf216, buf240, 1, 3072, grid=grid(1), stream=stream0)
        buf243 = empty_strided_cuda((4, 1, 70), (70, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_12.run(primals_4, buf243, 280, grid=grid(280), stream=stream0)
        buf244 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_50], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_79, buf244, 32, grid=grid(32), stream=stream0)
        buf278 = empty_strided_cuda((4, 1, 70), (70, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_12.run(primals_23, buf278, 280, grid=grid(280), stream=stream0)
        buf249 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf250 = reinterpret_tensor(buf249, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf249  # reuse
        buf251 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        buf283 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_51, _weight_norm_57], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_7.run(buf250, primals_82, primals_81, buf251, buf283, 128, 160, grid=grid(128), stream=stream0)
        buf255 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf256 = reinterpret_tensor(buf255, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf255  # reuse
        buf257 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        buf287 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_52, _weight_norm_58], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_8.run(buf256, primals_85, primals_84, buf257, buf287, 512, 640, grid=grid(512), stream=stream0)
        buf261 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf262 = reinterpret_tensor(buf261, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf261  # reuse
        buf263 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        buf291 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_53, _weight_norm_59], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf262, primals_88, primals_87, buf263, buf291, 1024, 2560, grid=grid(1024), stream=stream0)
        buf267 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf268 = reinterpret_tensor(buf267, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf267  # reuse
        buf269 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        buf295 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_54, _weight_norm_60], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_4.run(buf268, primals_91, primals_90, buf269, buf295, 1024, 5120, grid=grid(1024), stream=stream0)
        buf273 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf274 = buf273; del buf273  # reuse
        buf275 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        buf299 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_55, _weight_norm_61], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf274, primals_94, primals_93, buf275, buf299, 1, 3072, grid=grid(1), stream=stream0)
        buf302 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_62], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_6.run(primals_97, buf302, 32, grid=grid(32), stream=stream0)
        buf307 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf308 = reinterpret_tensor(buf307, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf307  # reuse
        buf309 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        buf340 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_63, _weight_norm_69], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_7.run(buf308, primals_100, primals_99, buf309, buf340, 128, 160, grid=grid(128), stream=stream0)
        buf313 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf314 = reinterpret_tensor(buf313, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf313  # reuse
        buf315 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        buf344 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_64, _weight_norm_70], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_8.run(buf314, primals_103, primals_102, buf315, buf344, 512, 640, grid=grid(512), stream=stream0)
        buf319 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf320 = reinterpret_tensor(buf319, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf319  # reuse
        buf321 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        buf348 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_65, _weight_norm_71], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf320, primals_106, primals_105, buf321, buf348, 1024, 2560, grid=grid(1024), stream=stream0)
        buf325 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf326 = reinterpret_tensor(buf325, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf325  # reuse
        buf327 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        buf352 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_66, _weight_norm_72], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_4.run(buf326, primals_109, primals_108, buf327, buf352, 1024, 5120, grid=grid(1024), stream=stream0)
        buf331 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf332 = buf331; del buf331  # reuse
        buf333 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        buf356 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_67, _weight_norm_73], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_5.run(buf332, primals_112, primals_111, buf333, buf356, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_4, buf2, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (4, 16, 64), (1024, 64, 1))
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(primals_23, buf41, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf42, (4, 16, 64), (1024, 64, 1))
        buf69 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        buf102 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_14, _weight_norm_20], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_25, primals_24, buf68, buf69, buf102, 160, grid=grid(160), stream=stream0)
        buf127 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        buf161 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_26, _weight_norm_32], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_43, primals_42, buf126, buf127, buf161, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(reinterpret_tensor(buf125, (4, 1, 22, 3), (66, 66, 3, 1), 0), buf127, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 32, 8, 3), (768, 24, 3, 1))
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(reinterpret_tensor(buf160, (4, 1, 22, 3), (66, 66, 3, 1), 0), buf161, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 32, 8, 3), (768, 24, 3, 1))
        buf186 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        buf220 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_38, _weight_norm_44], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_61, primals_60, buf185, buf186, buf220, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(reinterpret_tensor(buf184, (4, 1, 13, 5), (65, 0, 5, 1), 0), buf186, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 32, 5, 5), (800, 25, 5, 1))
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(reinterpret_tensor(buf219, (4, 1, 13, 5), (65, 0, 5, 1), 0), buf220, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 32, 5, 5), (800, 25, 5, 1))
        buf245 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        buf279 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_50, _weight_norm_56], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_79, primals_78, buf244, buf245, buf279, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(reinterpret_tensor(buf243, (4, 1, 10, 7), (70, 0, 7, 1), 0), buf245, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 32, 4, 7), (896, 28, 7, 1))
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(reinterpret_tensor(buf278, (4, 1, 10, 7), (70, 0, 7, 1), 0), buf279, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 32, 4, 7), (896, 28, 7, 1))
        buf303 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        buf336 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_weight_norm_62, _weight_norm_68], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_13.run(primals_97, primals_96, buf302, buf303, buf336, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(reinterpret_tensor(buf125, (4, 1, 6, 11), (66, 66, 11, 1), 0), buf303, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 32, 2, 11), (704, 22, 11, 1))
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(reinterpret_tensor(buf160, (4, 1, 6, 11), (66, 66, 11, 1), 0), buf336, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (4, 32, 2, 11), (704, 22, 11, 1))
        buf4 = empty_strided_cuda((4, 16, 64), (1024, 64, 1), torch.bool)
        buf5 = buf3; del buf3  # reuse
        buf43 = empty_strided_cuda((4, 16, 64), (1024, 64, 1), torch.bool)
        buf44 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_14, x_15], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_14.run(buf5, buf44, primals_3, buf4, buf43, 4096, grid=grid(4096), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf5, buf8, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf9, (4, 64, 16), (1024, 16, 1))
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf46, (4, 64, 16), (1024, 16, 1))
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(reinterpret_tensor(primals_4, (4, 1, 32, 2), (64, 64, 2, 1), 0), buf69, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 32, 11, 2), (704, 22, 2, 1))
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(reinterpret_tensor(primals_23, (4, 1, 32, 2), (64, 64, 2, 1), 0), buf102, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 32, 11, 2), (704, 22, 2, 1))
        buf129 = empty_strided_cuda((4, 32, 8, 3), (768, 24, 3, 1), torch.bool)
        buf130 = buf128; del buf128  # reuse
        buf163 = empty_strided_cuda((4, 32, 8, 3), (768, 24, 3, 1), torch.bool)
        buf164 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_56, x_57, x_70, x_71], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_15.run(buf130, buf164, primals_44, buf129, buf163, 3072, grid=grid(3072), stream=stream0)
        del primals_44
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf130, buf133, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 128, 3, 3), (1152, 9, 3, 1))
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf164, buf165, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 128, 3, 3), (1152, 9, 3, 1))
        buf188 = empty_strided_cuda((4, 32, 5, 5), (800, 25, 5, 1), torch.bool)
        buf189 = buf187; del buf187  # reuse
        buf222 = empty_strided_cuda((4, 32, 5, 5), (800, 25, 5, 1), torch.bool)
        buf223 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_84, x_85, x_98, x_99], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_16.run(buf189, buf223, primals_62, buf188, buf222, 3200, grid=grid(3200), stream=stream0)
        del primals_62
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf189, buf192, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 128, 2, 5), (1280, 10, 5, 1))
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf223, buf224, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 128, 2, 5), (1280, 10, 5, 1))
        buf247 = empty_strided_cuda((4, 32, 4, 7), (896, 28, 7, 1), torch.bool)
        buf248 = buf246; del buf246  # reuse
        buf281 = empty_strided_cuda((4, 32, 4, 7), (896, 28, 7, 1), torch.bool)
        buf282 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [x_112, x_113, x_126, x_127], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_17.run(buf248, buf282, primals_80, buf247, buf281, 3584, grid=grid(3584), stream=stream0)
        del primals_80
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf248, buf251, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 128, 2, 7), (1792, 14, 7, 1))
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf282, buf283, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 128, 2, 7), (1792, 14, 7, 1))
        buf305 = empty_strided_cuda((4, 32, 2, 11), (704, 22, 11, 1), torch.bool)
        buf306 = buf304; del buf304  # reuse
        buf338 = empty_strided_cuda((4, 32, 2, 11), (704, 22, 11, 1), torch.bool)
        buf339 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [x_140, x_141, x_154, x_155], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf306, buf339, primals_98, buf305, buf338, 2816, grid=grid(2816), stream=stream0)
        del primals_98
        # Topologically Sorted Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf306, buf309, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 128, 1, 11), (1408, 11, 11, 1))
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf339, buf340, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 128, 1, 11), (1408, 11, 11, 1))
        buf10 = empty_strided_cuda((4, 64, 16), (1024, 16, 1), torch.bool)
        buf11 = buf9; del buf9  # reuse
        buf47 = empty_strided_cuda((4, 64, 16), (1024, 16, 1), torch.bool)
        buf48 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, x_16, x_17], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_19.run(buf11, buf48, primals_7, buf10, buf47, 4096, grid=grid(4096), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf11, buf14, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf15, (4, 256, 4), (1024, 4, 1))
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf50, (4, 256, 4), (1024, 4, 1))
        buf71 = empty_strided_cuda((4, 32, 11, 2), (704, 22, 2, 1), torch.bool)
        buf72 = buf70; del buf70  # reuse
        buf104 = empty_strided_cuda((4, 32, 11, 2), (704, 22, 2, 1), torch.bool)
        buf105 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_29, x_30, x_42, x_43], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_18.run(buf72, buf105, primals_26, buf71, buf104, 2816, grid=grid(2816), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf72, buf75, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 128, 4, 2), (1024, 8, 2, 1))
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf105, buf106, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 128, 4, 2), (1024, 8, 2, 1))
        buf135 = empty_strided_cuda((4, 128, 3, 3), (1152, 9, 3, 1), torch.bool)
        buf136 = buf134; del buf134  # reuse
        buf167 = empty_strided_cuda((4, 128, 3, 3), (1152, 9, 3, 1), torch.bool)
        buf168 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_58, x_59, x_72, x_73], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_20.run(buf136, buf168, primals_47, buf135, buf167, 4608, grid=grid(4608), stream=stream0)
        del primals_47
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf136, buf139, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 512, 1, 3), (1536, 3, 3, 1))
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf168, buf169, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 512, 1, 3), (1536, 3, 3, 1))
        buf194 = empty_strided_cuda((4, 128, 2, 5), (1280, 10, 5, 1), torch.bool)
        buf195 = buf193; del buf193  # reuse
        buf226 = empty_strided_cuda((4, 128, 2, 5), (1280, 10, 5, 1), torch.bool)
        buf227 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_86, x_87, x_100, x_101], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_21.run(buf195, buf227, primals_65, buf194, buf226, 5120, grid=grid(5120), stream=stream0)
        del primals_65
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf195, buf198, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 512, 1, 5), (2560, 5, 5, 1))
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf227, buf228, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 512, 1, 5), (2560, 5, 5, 1))
        buf253 = empty_strided_cuda((4, 128, 2, 7), (1792, 14, 7, 1), torch.bool)
        buf254 = buf252; del buf252  # reuse
        buf285 = empty_strided_cuda((4, 128, 2, 7), (1792, 14, 7, 1), torch.bool)
        buf286 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [x_114, x_115, x_128, x_129], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_22.run(buf254, buf286, primals_83, buf253, buf285, 7168, grid=grid(7168), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf254, buf257, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 512, 1, 7), (3584, 7, 7, 1))
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf286, buf287, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 512, 1, 7), (3584, 7, 7, 1))
        buf311 = empty_strided_cuda((4, 128, 1, 11), (1408, 11, 11, 1), torch.bool)
        buf312 = buf310; del buf310  # reuse
        buf342 = empty_strided_cuda((4, 128, 1, 11), (1408, 11, 11, 1), torch.bool)
        buf343 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [x_142, x_143, x_156, x_157], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_23.run(buf312, buf343, primals_101, buf311, buf342, 5632, grid=grid(5632), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf312, buf315, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 512, 1, 11), (5632, 11, 11, 1))
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf343, buf344, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 512, 1, 11), (5632, 11, 11, 1))
        buf16 = empty_strided_cuda((4, 256, 4), (1024, 4, 1), torch.bool)
        buf17 = buf15; del buf15  # reuse
        buf51 = empty_strided_cuda((4, 256, 4), (1024, 4, 1), torch.bool)
        buf52 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_5, x_18, x_19], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_24.run(buf17, buf52, primals_10, buf16, buf51, 4096, grid=grid(4096), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf17, buf20, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=64, bias=None)
        assert_size_stride(buf21, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=64, bias=None)
        assert_size_stride(buf54, (4, 1024, 1), (1024, 1, 1))
        buf77 = empty_strided_cuda((4, 128, 4, 2), (1024, 8, 2, 1), torch.bool)
        buf78 = buf76; del buf76  # reuse
        buf108 = empty_strided_cuda((4, 128, 4, 2), (1024, 8, 2, 1), torch.bool)
        buf109 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_31, x_32, x_44, x_45], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_25.run(buf78, buf109, primals_29, buf77, buf108, 4096, grid=grid(4096), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf78, buf81, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf109, buf110, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf141 = empty_strided_cuda((4, 512, 1, 3), (1536, 3, 3, 1), torch.bool)
        buf142 = buf140; del buf140  # reuse
        buf171 = empty_strided_cuda((4, 512, 1, 3), (1536, 3, 3, 1), torch.bool)
        buf172 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_61, x_74, x_75], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_26.run(buf142, buf172, primals_50, buf141, buf171, 6144, grid=grid(6144), stream=stream0)
        del primals_50
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf142, buf145, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 1024, 1, 3), (3072, 3, 3, 1))
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf172, buf173, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 1024, 1, 3), (3072, 3, 3, 1))
        buf200 = empty_strided_cuda((4, 512, 1, 5), (2560, 5, 5, 1), torch.bool)
        buf201 = buf199; del buf199  # reuse
        buf230 = empty_strided_cuda((4, 512, 1, 5), (2560, 5, 5, 1), torch.bool)
        buf231 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [x_88, x_89, x_102, x_103], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_27.run(buf201, buf231, primals_68, buf200, buf230, 10240, grid=grid(10240), stream=stream0)
        del primals_68
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf201, buf204, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 1024, 1, 5), (5120, 5, 5, 1))
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf231, buf232, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 1024, 1, 5), (5120, 5, 5, 1))
        buf259 = empty_strided_cuda((4, 512, 1, 7), (3584, 7, 7, 1), torch.bool)
        buf260 = buf258; del buf258  # reuse
        buf289 = empty_strided_cuda((4, 512, 1, 7), (3584, 7, 7, 1), torch.bool)
        buf290 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_116, x_117, x_130, x_131], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_28.run(buf260, buf290, primals_86, buf259, buf289, 14336, grid=grid(14336), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf260, buf263, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 1024, 1, 7), (7168, 7, 7, 1))
        # Topologically Sorted Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf290, buf291, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 1024, 1, 7), (7168, 7, 7, 1))
        buf317 = empty_strided_cuda((4, 512, 1, 11), (5632, 11, 11, 1), torch.bool)
        buf318 = buf316; del buf316  # reuse
        buf346 = empty_strided_cuda((4, 512, 1, 11), (5632, 11, 11, 1), torch.bool)
        buf347 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_144, x_145, x_158, x_159], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_29.run(buf318, buf347, primals_104, buf317, buf346, 22528, grid=grid(22528), stream=stream0)
        del primals_104
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf318, buf321, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 1024, 1, 11), (11264, 11, 11, 1))
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf347, buf348, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 1024, 1, 11), (11264, 11, 11, 1))
        buf22 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf23 = buf21; del buf21  # reuse
        buf55 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf56 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7, x_20, x_21], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_30.run(buf23, buf56, primals_13, buf22, buf55, 4096, grid=grid(4096), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf23, buf26, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=256, bias=None)
        assert_size_stride(buf27, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf56, buf57, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=256, bias=None)
        assert_size_stride(buf58, (4, 1024, 1), (1024, 1, 1))
        buf83 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.bool)
        buf84 = buf82; del buf82  # reuse
        buf112 = empty_strided_cuda((4, 512, 2, 2), (2048, 4, 2, 1), torch.bool)
        buf113 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_33, x_34, x_46, x_47], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_31.run(buf84, buf113, primals_32, buf83, buf112, 8192, grid=grid(8192), stream=stream0)
        del primals_32
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf84, buf87, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 1024, 1, 2), (2048, 2, 2, 1))
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf113, buf114, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 1024, 1, 2), (2048, 2, 2, 1))
        buf147 = empty_strided_cuda((4, 1024, 1, 3), (3072, 3, 3, 1), torch.bool)
        buf148 = buf146; del buf146  # reuse
        buf175 = empty_strided_cuda((4, 1024, 1, 3), (3072, 3, 3, 1), torch.bool)
        buf176 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_62, x_63, x_76, x_77], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_32.run(buf148, buf176, primals_53, buf147, buf175, 12288, grid=grid(12288), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf148, buf151, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 1024, 1, 3), (3072, 3, 3, 1))
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 1024, 1, 3), (3072, 3, 3, 1))
        buf206 = empty_strided_cuda((4, 1024, 1, 5), (5120, 5, 5, 1), torch.bool)
        buf207 = buf205; del buf205  # reuse
        buf234 = empty_strided_cuda((4, 1024, 1, 5), (5120, 5, 5, 1), torch.bool)
        buf235 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_90, x_91, x_104, x_105], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_33.run(buf207, buf235, primals_71, buf206, buf234, 20480, grid=grid(20480), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf207, buf210, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 1024, 1, 5), (5120, 5, 5, 1))
        # Topologically Sorted Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf235, buf236, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 1024, 1, 5), (5120, 5, 5, 1))
        buf265 = empty_strided_cuda((4, 1024, 1, 7), (7168, 7, 7, 1), torch.bool)
        buf266 = buf264; del buf264  # reuse
        buf293 = empty_strided_cuda((4, 1024, 1, 7), (7168, 7, 7, 1), torch.bool)
        buf294 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [x_118, x_119, x_132, x_133], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_34.run(buf266, buf294, primals_89, buf265, buf293, 28672, grid=grid(28672), stream=stream0)
        del primals_89
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf266, buf269, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 1024, 1, 7), (7168, 7, 7, 1))
        # Topologically Sorted Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf294, buf295, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 1024, 1, 7), (7168, 7, 7, 1))
        buf323 = empty_strided_cuda((4, 1024, 1, 11), (11264, 11, 11, 1), torch.bool)
        buf324 = buf322; del buf322  # reuse
        buf350 = empty_strided_cuda((4, 1024, 1, 11), (11264, 11, 11, 1), torch.bool)
        buf351 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [x_146, x_147, x_160, x_161], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_35.run(buf324, buf351, primals_107, buf323, buf350, 45056, grid=grid(45056), stream=stream0)
        del primals_107
        # Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf324, buf327, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 1024, 1, 11), (11264, 11, 11, 1))
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf351, buf352, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 1024, 1, 11), (11264, 11, 11, 1))
        buf28 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf29 = buf27; del buf27  # reuse
        buf59 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf60 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_8, x_9, x_22, x_23], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_30.run(buf29, buf60, primals_16, buf28, buf59, 4096, grid=grid(4096), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf29, buf32, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf33, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf62, (4, 1024, 1), (1024, 1, 1))
        buf89 = empty_strided_cuda((4, 1024, 1, 2), (2048, 2, 2, 1), torch.bool)
        buf90 = buf88; del buf88  # reuse
        buf116 = empty_strided_cuda((4, 1024, 1, 2), (2048, 2, 2, 1), torch.bool)
        buf117 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_35, x_36, x_48, x_49], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_36.run(buf90, buf117, primals_35, buf89, buf116, 8192, grid=grid(8192), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf90, buf93, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 1024, 1, 2), (2048, 2, 2, 1))
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 1024, 1, 2), (2048, 2, 2, 1))
        buf153 = empty_strided_cuda((4, 1024, 1, 3), (3072, 3, 3, 1), torch.bool)
        buf154 = buf152; del buf152  # reuse
        buf179 = empty_strided_cuda((4, 1024, 1, 3), (3072, 3, 3, 1), torch.bool)
        buf180 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_64, x_65, x_78, x_79], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_32.run(buf154, buf180, primals_56, buf153, buf179, 12288, grid=grid(12288), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf154, buf157, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 1, 1, 3), (3, 3, 3, 1))
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 1, 1, 3), (3, 3, 3, 1))
        buf212 = empty_strided_cuda((4, 1024, 1, 5), (5120, 5, 5, 1), torch.bool)
        buf213 = buf211; del buf211  # reuse
        buf238 = empty_strided_cuda((4, 1024, 1, 5), (5120, 5, 5, 1), torch.bool)
        buf239 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [x_92, x_93, x_106, x_107], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_33.run(buf213, buf239, primals_74, buf212, buf238, 20480, grid=grid(20480), stream=stream0)
        del primals_74
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf213, buf216, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 1, 1, 5), (5, 5, 5, 1))
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf239, buf240, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 1, 1, 5), (5, 5, 5, 1))
        buf271 = empty_strided_cuda((4, 1024, 1, 7), (7168, 7, 7, 1), torch.bool)
        buf272 = buf270; del buf270  # reuse
        buf297 = empty_strided_cuda((4, 1024, 1, 7), (7168, 7, 7, 1), torch.bool)
        buf298 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_120, x_121, x_134, x_135], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_34.run(buf272, buf298, primals_92, buf271, buf297, 28672, grid=grid(28672), stream=stream0)
        del primals_92
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf272, buf275, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 1, 1, 7), (7, 7, 7, 1))
        # Topologically Sorted Source Nodes: [x_136], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf298, buf299, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 1, 1, 7), (7, 7, 7, 1))
        buf329 = empty_strided_cuda((4, 1024, 1, 11), (11264, 11, 11, 1), torch.bool)
        buf330 = buf328; del buf328  # reuse
        buf354 = empty_strided_cuda((4, 1024, 1, 11), (11264, 11, 11, 1), torch.bool)
        buf355 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [x_148, x_149, x_162, x_163], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_35.run(buf330, buf355, primals_110, buf329, buf354, 45056, grid=grid(45056), stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf330, buf333, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 1, 1, 11), (11, 11, 11, 1))
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf355, buf356, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (4, 1, 1, 11), (11, 11, 11, 1))
        buf34 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf35 = buf33; del buf33  # reuse
        buf63 = empty_strided_cuda((4, 1024, 1), (1024, 1, 1), torch.bool)
        buf64 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_10, x_11, x_24, x_25], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_30.run(buf35, buf64, primals_19, buf34, buf63, 4096, grid=grid(4096), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf35, buf38, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf39, (4, 1, 1), (1, 1, 1))
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf66, (4, 1, 1), (1, 1, 1))
        buf95 = empty_strided_cuda((4, 1024, 1, 2), (2048, 2, 2, 1), torch.bool)
        buf96 = buf94; del buf94  # reuse
        buf120 = empty_strided_cuda((4, 1024, 1, 2), (2048, 2, 2, 1), torch.bool)
        buf121 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_37, x_38, x_50, x_51], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_36.run(buf96, buf121, primals_38, buf95, buf120, 8192, grid=grid(8192), stream=stream0)
        del primals_38
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf96, buf99, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 1, 1, 2), (2, 2, 2, 1))
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf121, buf122, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 1, 1, 2), (2, 2, 2, 1))
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf159, primals_59, 12, grid=grid(12), stream=stream0)
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf183, primals_59, 12, grid=grid(12), stream=stream0)
        del primals_59
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf218, primals_77, 20, grid=grid(20), stream=stream0)
        buf242 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf242, primals_77, 20, grid=grid(20), stream=stream0)
        del primals_77
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_39.run(buf277, primals_95, 28, grid=grid(28), stream=stream0)
        buf301 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [x_136], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_39.run(buf301, primals_95, 28, grid=grid(28), stream=stream0)
        del primals_95
        buf335 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_40.run(buf335, primals_113, 44, grid=grid(44), stream=stream0)
        buf358 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_40.run(buf358, primals_113, 44, grid=grid(44), stream=stream0)
        del primals_113
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_41.run(buf40, primals_22, 4, grid=grid(4), stream=stream0)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_41.run(buf67, primals_22, 4, grid=grid(4), stream=stream0)
        del primals_22
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf101, primals_41, 8, grid=grid(8), stream=stream0)
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf124, primals_41, 8, grid=grid(8), stream=stream0)
        del primals_41
    return (reinterpret_tensor(buf40, (4, 1), (1, 1), 0), reinterpret_tensor(buf101, (4, 2), (2, 1), 0), reinterpret_tensor(buf159, (4, 3), (3, 1), 0), reinterpret_tensor(buf218, (4, 5), (5, 1), 0), reinterpret_tensor(buf277, (4, 7), (7, 1), 0), reinterpret_tensor(buf335, (4, 11), (11, 1), 0), reinterpret_tensor(buf67, (4, 1), (1, 1), 0), reinterpret_tensor(buf124, (4, 2), (2, 1), 0), reinterpret_tensor(buf183, (4, 3), (3, 1), 0), reinterpret_tensor(buf242, (4, 5), (5, 1), 0), reinterpret_tensor(buf301, (4, 7), (7, 1), 0), reinterpret_tensor(buf358, (4, 11), (11, 1), 0), buf5, buf11, buf17, buf23, buf29, buf35, buf40, buf72, buf78, buf84, buf90, buf96, buf101, buf130, buf136, buf142, buf148, buf154, buf159, buf189, buf195, buf201, buf207, buf213, buf218, buf248, buf254, buf260, buf266, buf272, buf277, buf306, buf312, buf318, buf324, buf330, buf335, buf44, buf48, buf52, buf56, buf60, buf64, buf67, buf105, buf109, buf113, buf117, buf121, buf124, buf164, buf168, buf172, buf176, buf180, buf183, buf223, buf227, buf231, buf235, buf239, buf242, buf282, buf286, buf290, buf294, buf298, buf301, buf339, buf343, buf347, buf351, buf355, buf358, buf41, buf45, buf49, buf53, buf57, buf61, buf65, buf102, buf106, buf110, buf114, buf118, buf122, buf161, buf165, buf169, buf173, buf177, buf181, buf220, buf224, buf228, buf232, buf236, buf240, buf279, buf283, buf287, buf291, buf295, buf299, buf336, buf340, buf344, buf348, buf352, buf356, primals_1, primals_2, primals_4, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, buf22, buf23, buf25, buf26, buf28, buf29, buf31, buf32, buf34, buf35, buf37, buf38, buf41, buf43, buf44, buf45, buf47, buf48, buf49, buf51, buf52, buf53, buf55, buf56, buf57, buf59, buf60, buf61, buf63, buf64, buf65, buf68, buf69, buf71, buf72, buf74, buf75, buf77, buf78, buf80, buf81, buf83, buf84, buf86, buf87, buf89, buf90, buf92, buf93, buf95, buf96, buf98, buf99, buf102, buf104, buf105, buf106, buf108, buf109, buf110, buf112, buf113, buf114, buf116, buf117, buf118, buf120, buf121, buf122, buf125, buf126, buf127, buf129, buf130, buf132, buf133, buf135, buf136, buf138, buf139, buf141, buf142, buf144, buf145, buf147, buf148, buf150, buf151, buf153, buf154, buf156, buf157, buf160, buf161, buf163, buf164, buf165, buf167, buf168, buf169, buf171, buf172, buf173, buf175, buf176, buf177, buf179, buf180, buf181, reinterpret_tensor(buf184, (4, 1, 13, 5), (65, 65, 5, 1), 0), buf185, buf186, buf188, buf189, buf191, buf192, buf194, buf195, buf197, buf198, buf200, buf201, buf203, buf204, buf206, buf207, buf209, buf210, buf212, buf213, buf215, buf216, reinterpret_tensor(buf219, (4, 1, 13, 5), (65, 65, 5, 1), 0), buf220, buf222, buf223, buf224, buf226, buf227, buf228, buf230, buf231, buf232, buf234, buf235, buf236, buf238, buf239, buf240, reinterpret_tensor(buf243, (4, 1, 10, 7), (70, 70, 7, 1), 0), buf244, buf245, buf247, buf248, buf250, buf251, buf253, buf254, buf256, buf257, buf259, buf260, buf262, buf263, buf265, buf266, buf268, buf269, buf271, buf272, buf274, buf275, reinterpret_tensor(buf278, (4, 1, 10, 7), (70, 70, 7, 1), 0), buf279, buf281, buf282, buf283, buf285, buf286, buf287, buf289, buf290, buf291, buf293, buf294, buf295, buf297, buf298, buf299, buf302, buf303, buf305, buf306, buf308, buf309, buf311, buf312, buf314, buf315, buf317, buf318, buf320, buf321, buf323, buf324, buf326, buf327, buf329, buf330, buf332, buf333, buf336, buf338, buf339, buf340, buf342, buf343, buf344, buf346, buf347, buf348, buf350, buf351, buf352, buf354, buf355, buf356, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1024, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
