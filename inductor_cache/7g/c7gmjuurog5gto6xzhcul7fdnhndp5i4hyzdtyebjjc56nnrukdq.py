# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/37/c37npq3cl6yjhvnh2wf4mw3mp6ejvlpwhqswwo6ildu42e3tx5np.py
# Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, mul, output], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   mean => mean
#   mul => mul
#   output => mul_1
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 4), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_2), kwargs = {})
triton_poi_fused_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_poi_fused_add_mean_mul_pow_rsqrt_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp13 + tmp12
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp0 * tmp15
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vh/cvh6cfwwr3preimj7qozxjfj4y4bq4l7jmzcekwrqcjwoc3ggowj.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attn_output => clone, mul_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_3, 1.0), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_1 = async_compile.triton('triton_poi_fused_clone_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 4*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/nk/cnkl74nz5qg7yewaptikyxoz5tl3l7okz5jqszrni4ys6wkrnb3d.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.ones, aten.tril, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   attn_output => add_1, amax, any_1, eq, exp, full_default, full_default_1, full_default_2, le, logical_and, logical_not, sub, sub_1, sum_1, where
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %full_default_2, %full_default_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %where), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_1, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_1, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2 = async_compile.triton('triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = (-1)*x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = 0.0
    tmp7 = float("-inf")
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tmp11 = 1 + ((-1)*x0)
    tmp12 = tmp11 <= tmp2
    tmp13 = tmp12 & tmp4
    tmp14 = tl.where(tmp13, tmp6, tmp7)
    tmp15 = tmp10 + tmp14
    tmp16 = triton_helpers.maximum(tmp9, tmp15)
    tmp18 = 2 + ((-1)*x0)
    tmp19 = tmp18 <= tmp2
    tmp20 = tmp19 & tmp4
    tmp21 = tl.where(tmp20, tmp6, tmp7)
    tmp22 = tmp17 + tmp21
    tmp23 = triton_helpers.maximum(tmp16, tmp22)
    tmp25 = 3 + ((-1)*x0)
    tmp26 = tmp25 <= tmp2
    tmp27 = tmp26 & tmp4
    tmp28 = tl.where(tmp27, tmp6, tmp7)
    tmp29 = tmp24 + tmp28
    tmp30 = triton_helpers.maximum(tmp23, tmp29)
    tmp31 = tmp9 - tmp30
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp15 - tmp30
    tmp34 = tl_math.exp(tmp33)
    tmp35 = tmp32 + tmp34
    tmp36 = tmp22 - tmp30
    tmp37 = tl_math.exp(tmp36)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp29 - tmp30
    tmp40 = tl_math.exp(tmp39)
    tmp41 = tmp38 + tmp40
    tmp42 = tmp9 == tmp7
    tmp43 = tmp42 == 0
    tmp44 = tmp43.to(tl.int64)
    tmp45 = (tmp44 != 0)
    tmp46 = tmp15 == tmp7
    tmp47 = tmp46 == 0
    tmp48 = tmp47.to(tl.int64)
    tmp49 = (tmp48 != 0)
    tmp50 = tmp45 | tmp49
    tmp51 = tmp22 == tmp7
    tmp52 = tmp51 == 0
    tmp53 = tmp52.to(tl.int64)
    tmp54 = (tmp53 != 0)
    tmp55 = tmp50 | tmp54
    tmp56 = tmp29 == tmp7
    tmp57 = tmp56 == 0
    tmp58 = tmp57.to(tl.int64)
    tmp59 = (tmp58 != 0)
    tmp60 = tmp55 | tmp59
    tl.store(out_ptr0 + (x2), tmp30, xmask)
    tl.store(out_ptr1 + (x2), tmp41, xmask)
    tl.store(out_ptr2 + (x2), tmp60, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ec/cechcneqzunbx7o65ys3gj2wq34xytcveumyvrn5covwlgt2gy7t.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.ones, aten.tril, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   attn_output => add_1, div, exp, full_default, full_default_1, full_default_2, full_default_3, le, logical_and, logical_not_1, sub, sub_1, where, where_1
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %full_default_2, %full_default_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %where), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default_3, %div), kwargs = {})
triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3 = async_compile.triton('triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 4
    x4 = xindex
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp12 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp3 = x0 + ((-1)*x1)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1], True, tl.int1)
    tmp7 = tmp5 & tmp6
    tmp8 = 0.0
    tmp9 = float("-inf")
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tmp13 = tmp11 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp16 = tmp14 / tmp15
    tmp17 = tl.where(tmp1, tmp8, tmp16)
    tl.store(in_out_ptr0 + (x4), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/an/cany2jbw2jdajpiihiclpvumcasljofi46nmewpzgmcfjpkuy5qh.py
# Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/z2/cz2mwn2hidmpauodbuq7nfkemjwa5lzw23if74vok5kgkgtfxaim.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x => add_2
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_17), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_poi_fused_add_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, mul, output], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_mul_pow_rsqrt_0.run(primals_1, primals_2, buf0, 64, grid=grid(64), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [queries], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (16, 4), (4, 1), 0), reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf1)
        buf2 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [keys], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (16, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf2)
        buf3 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [values], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 4), (1, 4), 0), out=buf3)
        buf4 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_1.run(buf1, buf4, 16, 4, grid=grid(16, 4), stream=stream0)
        buf5 = reinterpret_tensor(buf1, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_1.run(buf2, buf5, 16, 4, grid=grid(16, 4), stream=stream0)
        buf6 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf5, (16, 1, 4), (4, 0, 1), 0), out=buf6)
        buf7 = reinterpret_tensor(buf2, (4, 4, 4, 1), (16, 4, 1, 64), 0); del buf2  # reuse
        buf8 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        buf9 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.bool)
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.ones, aten.tril, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2.run(buf6, buf7, buf8, buf9, 64, grid=grid(64), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.ones, aten.tril, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_3.run(buf10, buf9, buf7, buf8, 256, grid=grid(256), stream=stream0)
        del buf9
        buf11 = reinterpret_tensor(buf8, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf3, buf11, 16, 4, grid=grid(16, 4), stream=stream0)
        buf12 = reinterpret_tensor(buf3, (16, 4, 1), (4, 1, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf11, (16, 4, 1), (4, 1, 0), 0), out=buf12)
        buf13 = reinterpret_tensor(buf7, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf12, buf13, 16, 4, grid=grid(16, 4), stream=stream0)
        buf14 = reinterpret_tensor(buf12, (16, 4), (4, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (16, 4), (4, 1), 0), reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf14)
        buf15 = reinterpret_tensor(buf14, (4, 4, 4), (16, 4, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf15, primals_1, 64, grid=grid(64), stream=stream0)
        buf16 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_2, mean_1, add_2, rsqrt_1, mul_2, output_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_mul_pow_rsqrt_0.run(buf15, primals_7, buf16, 64, grid=grid(64), stream=stream0)
    return (buf16, buf15, primals_1, primals_7, reinterpret_tensor(buf0, (16, 4), (4, 1), 0), buf10, reinterpret_tensor(buf13, (16, 4), (4, 1), 0), buf15, primals_6, reinterpret_tensor(buf11, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf4, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf5, (16, 4, 1), (4, 1, 4), 0), primals_5, primals_4, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
