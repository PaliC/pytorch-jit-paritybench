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


# kernel path: inductor_cache/o5/co5dc4oj6dccp7qvt4emwciwmk5ek3jvtlf3btj6ywkzmrxjpzrt.py
# Topologically Sorted Source Nodes: [norm, x_norm, x_norm_1], Original ATen: [aten.linalg_vector_norm, aten.clamp, aten.div]
# Source node to ATen node mapping:
#   norm => pow_1, pow_2, sum_1
#   x_norm => clamp_min
#   x_norm_1 => div
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-09), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %clamp_min), kwargs = {})
triton_poi_fused_clamp_div_linalg_vector_norm_0 = async_compile.triton('triton_poi_fused_clamp_div_linalg_vector_norm_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_linalg_vector_norm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_linalg_vector_norm_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-09
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/et/ceti2xy54tuyjjujbgechfky3kle35iwvo6pniapgignk7s7gqew.py
# Topologically Sorted Source Nodes: [norm_1, w_norm, w_norm_1], Original ATen: [aten.linalg_vector_norm, aten.clamp, aten.div]
# Source node to ATen node mapping:
#   norm_1 => pow_3, pow_4, sum_2
#   w_norm => clamp_min_1
#   w_norm_1 => div_1
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_3, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [0], True), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-09), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_3, %clamp_min_1), kwargs = {})
triton_poi_fused_clamp_div_linalg_vector_norm_1 = async_compile.triton('triton_poi_fused_clamp_div_linalg_vector_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_linalg_vector_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_linalg_vector_norm_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 10)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (10 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (20 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (30 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-09
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aa/caahordjbna6z4ry4d6fcd53s2e6cyzj7lj2vs5zvdpi4zkpenjw.py
# Topologically Sorted Source Nodes: [zeros_like, delt_costh], Original ATen: [aten.zeros_like, aten.scatter]
# Source node to ATen node mapping:
#   delt_costh => scatter
#   zeros_like => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([4, 10], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default, 1, %unsqueeze, 0.3), kwargs = {})
triton_poi_fused_scatter_zeros_like_2 = async_compile.triton('triton_poi_fused_scatter_zeros_like_2', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_zeros_like_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scatter_zeros_like_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ef/cefkx57cnui7l55f4fgrq3rlzrubwsq7pfxujakaqjfwevysxs2m.py
# Topologically Sorted Source Nodes: [zeros_like, delt_costh, loss], Original ATen: [aten.zeros_like, aten.scatter, aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   delt_costh => scatter
#   loss => full_default_1
#   zeros_like => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([4, 10], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default, 1, %unsqueeze, 0.3), kwargs = {})
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_3 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze, -100), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze, %full_default_1), kwargs = {})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default, 1, %where_2, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_nll_loss_forward_scatter_zeros_like_3 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_scatter_zeros_like_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_scatter_zeros_like_3', 'mutated_arg_names': ['out_ptr0', 'out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_scatter_zeros_like_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 10)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 10")
    tmp2 = 0.3
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp0 != tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.where(tmp4, tmp0, tmp5)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 10)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 10")
    tmp8 = -1.0
    tl.store(out_ptr0 + (tmp0 + 10*x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (tmp6 + 10*x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3r/c3rh7wes4kwhfhbqbhbjyh3qvnjowtkql5q2jzwpfngavuortajj.py
# Topologically Sorted Source Nodes: [costh_m, loss], Original ATen: [aten.sub, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   costh_m => sub
#   loss => exp, log, sub_2, sum_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm, %scatter), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 15), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_1,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_3,), kwargs = {})
#   %sub_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %log), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
triton_per_fused__log_softmax__log_softmax_backward_data_sub_4 = async_compile.triton('triton_per_fused__log_softmax__log_softmax_backward_data_sub_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__log_softmax_backward_data_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__log_softmax_backward_data_sub_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 10*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 10*x0), rmask & xmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = 15.0
    tmp11 = tmp9 * tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl_math.log(tmp16)
    tmp18 = tmp11 - tmp17
    tmp19 = tl_math.exp(tmp18)
    tl.store(out_ptr2 + (r1 + 10*x0), tmp19, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oy/coyglrdhx563abk7i7ycv2bstmntolyemjcvrp5cqttefrlvrkaz.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div_2, full_default_2, ne, neg, sum_4, sum_5, where_1
# Graph fragment:
#   %ne : [num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_2, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_4, torch.float32), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_5, %convert_element_type), kwargs = {})
triton_poi_fused_nll_loss_forward_5 = async_compile.triton('triton_poi_fused_nll_loss_forward_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_forward_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_forward_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp33 = tl.load(in_ptr3 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp38 = tl.load(in_ptr4 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp54 = tl.load(in_ptr3 + (1))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp58 = tl.load(in_ptr4 + (1))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp74 = tl.load(in_ptr3 + (2))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp78 = tl.load(in_ptr4 + (2))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp94 = tl.load(in_ptr3 + (3))
    tmp95 = tl.broadcast_to(tmp94, [XBLOCK])
    tmp98 = tl.load(in_ptr4 + (3))
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK])
    tmp2 = tl.full([1], -100, tl.int64)
    tmp3 = tmp1 != tmp2
    tmp4 = tmp3.to(tl.int64)
    tmp7 = tmp6 != tmp2
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp4 + tmp8
    tmp12 = tmp11 != tmp2
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tmp9 + tmp13
    tmp17 = tmp16 != tmp2
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tmp14 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.full([1], 0, tl.int64)
    tmp22 = tl.where(tmp3, tmp1, tmp21)
    tmp23 = tl.full([XBLOCK], 10, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tl.device_assert((0 <= tmp26) & (tmp26 < 10), "index out of bounds: 0 <= tmp26 < 10")
    tmp28 = tl.load(in_ptr1 + (tmp26), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr2 + (tmp26), None, eviction_policy='evict_last')
    tmp30 = tmp28 - tmp29
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp35 = tmp32 - tmp34
    tmp36 = 15.0
    tmp37 = tmp35 * tmp36
    tmp40 = tl_math.log(tmp39)
    tmp41 = tmp37 - tmp40
    tmp42 = -tmp41
    tmp43 = 0.0
    tmp44 = tl.where(tmp3, tmp42, tmp43)
    tmp45 = tl.where(tmp7, tmp6, tmp21)
    tmp46 = tmp45 + tmp23
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tl.device_assert((0 <= tmp48) & (tmp48 < 10), "index out of bounds: 0 <= tmp48 < 10")
    tmp50 = tl.load(in_ptr1 + (10 + tmp48), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr2 + (10 + tmp48), None, eviction_policy='evict_last')
    tmp52 = tmp50 - tmp51
    tmp53 = tmp52 * tmp31
    tmp56 = tmp53 - tmp55
    tmp57 = tmp56 * tmp36
    tmp60 = tl_math.log(tmp59)
    tmp61 = tmp57 - tmp60
    tmp62 = -tmp61
    tmp63 = tl.where(tmp7, tmp62, tmp43)
    tmp64 = tmp44 + tmp63
    tmp65 = tl.where(tmp12, tmp11, tmp21)
    tmp66 = tmp65 + tmp23
    tmp67 = tmp65 < 0
    tmp68 = tl.where(tmp67, tmp66, tmp65)
    tl.device_assert((0 <= tmp68) & (tmp68 < 10), "index out of bounds: 0 <= tmp68 < 10")
    tmp70 = tl.load(in_ptr1 + (20 + tmp68), None, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr2 + (20 + tmp68), None, eviction_policy='evict_last')
    tmp72 = tmp70 - tmp71
    tmp73 = tmp72 * tmp31
    tmp76 = tmp73 - tmp75
    tmp77 = tmp76 * tmp36
    tmp80 = tl_math.log(tmp79)
    tmp81 = tmp77 - tmp80
    tmp82 = -tmp81
    tmp83 = tl.where(tmp12, tmp82, tmp43)
    tmp84 = tmp64 + tmp83
    tmp85 = tl.where(tmp17, tmp16, tmp21)
    tmp86 = tmp85 + tmp23
    tmp87 = tmp85 < 0
    tmp88 = tl.where(tmp87, tmp86, tmp85)
    tl.device_assert((0 <= tmp88) & (tmp88 < 10), "index out of bounds: 0 <= tmp88 < 10")
    tmp90 = tl.load(in_ptr1 + (30 + tmp88), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr2 + (30 + tmp88), None, eviction_policy='evict_last')
    tmp92 = tmp90 - tmp91
    tmp93 = tmp92 * tmp31
    tmp96 = tmp93 - tmp95
    tmp97 = tmp96 * tmp36
    tmp100 = tl_math.log(tmp99)
    tmp101 = tmp97 - tmp100
    tmp102 = -tmp101
    tmp103 = tl.where(tmp17, tmp102, tmp43)
    tmp104 = tmp84 + tmp103
    tmp105 = tmp104 / tmp20
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp20, None)
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp105, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm, x_norm, x_norm_1], Original ATen: [aten.linalg_vector_norm, aten.clamp, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_div_linalg_vector_norm_0.run(primals_1, buf0, 16, grid=grid(16), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm_1, w_norm, w_norm_1], Original ATen: [aten.linalg_vector_norm, aten.clamp, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_div_linalg_vector_norm_1.run(primals_3, buf1, 40, grid=grid(40), stream=stream0)
        buf2 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm_1, w_norm, w_norm_1, costh], Original ATen: [aten.linalg_vector_norm, aten.clamp, aten.div, aten.mm]
        extern_kernels.mm(buf0, buf1, out=buf2)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [zeros_like, delt_costh], Original ATen: [aten.zeros_like, aten.scatter]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_zeros_like_2.run(buf3, 40, grid=grid(40), stream=stream0)
        buf10 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [zeros_like, loss], Original ATen: [aten.zeros_like, aten.nll_loss_forward, aten.nll_loss_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scatter_zeros_like_2.run(buf10, 40, grid=grid(40), stream=stream0)
        buf9 = empty_strided_cuda((4, 1), (1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [zeros_like, delt_costh, loss], Original ATen: [aten.zeros_like, aten.scatter, aten.nll_loss_forward, aten.nll_loss_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_scatter_zeros_like_3.run(primals_2, buf3, buf9, buf10, 4, grid=grid(4), stream=stream0)
        buf5 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf6 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf12 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [costh_m, loss], Original ATen: [aten.sub, aten._log_softmax, aten._log_softmax_backward_data]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__log_softmax_backward_data_sub_4.run(buf2, buf3, buf5, buf6, buf12, 4, 10, grid=grid(4), stream=stream0)
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf8 = empty_strided_cuda((), (), torch.float32)
        buf13 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_forward_5.run(buf13, primals_2, buf2, buf3, buf5, buf6, buf7, 1, grid=grid(1), stream=stream0)
        del buf2
        del buf3
        del buf5
        del buf6
        del primals_2
    return (buf13, primals_3, buf7, buf9, buf10, buf12, reinterpret_tensor(buf0, (4, 4), (1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((4, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
