# AOT ID: ['4_inference']
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


# kernel path: inductor_cache/of/cofms27qd34ltiqnm6zkv65iu6nghia2364ltq42slioffn5zmn3.py
# Topologically Sorted Source Nodes: [binary_cross_entropy_with_logits, mean], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mean]
# Source node to ATen node mapping:
#   binary_cross_entropy_with_logits => abs_1, exp, full_default, log1p, minimum, mul, neg, sub, sub_1, sub_2
#   mean => mean
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %view), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %permute), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %permute), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%permute,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %sub_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_2, [-1]), kwargs = {})
triton_poi_fused_binary_cross_entropy_with_logits_mean_0 = async_compile.triton('triton_poi_fused_binary_cross_entropy_with_logits_mean_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_binary_cross_entropy_with_logits_mean_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_binary_cross_entropy_with_logits_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.minimum(tmp5, tmp3)
    tmp7 = tl_math.abs(tmp3)
    tmp8 = -tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tmp6 - tmp10
    tmp12 = tmp4 - tmp11
    tmp14 = tmp1 - tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = triton_helpers.minimum(tmp5, tmp15)
    tmp18 = tl_math.abs(tmp15)
    tmp19 = -tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = libdevice.log1p(tmp20)
    tmp22 = tmp17 - tmp21
    tmp23 = tmp16 - tmp22
    tmp24 = tmp12 + tmp23
    tmp26 = tmp1 - tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = triton_helpers.minimum(tmp5, tmp27)
    tmp30 = tl_math.abs(tmp27)
    tmp31 = -tmp30
    tmp32 = tl_math.exp(tmp31)
    tmp33 = libdevice.log1p(tmp32)
    tmp34 = tmp29 - tmp33
    tmp35 = tmp28 - tmp34
    tmp36 = tmp24 + tmp35
    tmp38 = tmp1 - tmp37
    tmp40 = tmp38 * tmp39
    tmp41 = triton_helpers.minimum(tmp5, tmp39)
    tmp42 = tl_math.abs(tmp39)
    tmp43 = -tmp42
    tmp44 = tl_math.exp(tmp43)
    tmp45 = libdevice.log1p(tmp44)
    tmp46 = tmp41 - tmp45
    tmp47 = tmp40 - tmp46
    tmp48 = tmp36 + tmp47
    tmp49 = 4.0
    tmp50 = tmp48 / tmp49
    tl.store(out_ptr0 + (x2), tmp50, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jl/cjlu7hfux5orinwcir7yarxfqm7cewgfo4hywum3lofjidy5csg4.py
# Topologically Sorted Source Nodes: [loss_all_case, min_idx], Original ATen: [aten.mean, aten.argmin]
# Source node to ATen node mapping:
#   loss_all_case => mean_1
#   min_idx => argmin
# Graph fragment:
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mean, [1]), kwargs = {})
#   %argmin : [num_users=2] = call_function[target=torch.ops.aten.argmin.default](args = (%mean_1, -1), kwargs = {})
triton_per_fused_argmin_mean_1 = async_compile.triton('triton_per_fused_argmin_mean_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_argmin_mean_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_argmin_mean_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (96 + r1 + 384*x0), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (192 + r1 + 384*x0), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr0 + (288 + r1 + 384*x0), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("inf"))
    tmp12 = tl.broadcast_to(rindex, tmp11.shape)
    tmp10_val, tmp10_idx = triton_helpers.min_with_index(tmp11, tmp12, 1)
    tmp10 = tmp10_idx[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hp/chp5k3txe2trrpubqsg3cbbnjncowrw7zydfzf6zhn7sgf446sqt.py
# Topologically Sorted Source Nodes: [loss_all_case, selected_loss, sum_1, pit_min_loss], Original ATen: [aten.mean, aten.index, aten.sum, aten.div]
# Source node to ATen node mapping:
#   loss_all_case => mean_1
#   pit_min_loss => div
#   selected_loss => index_1
#   sum_1 => sum_1
# Graph fragment:
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mean, [1]), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mean_1, [%iota_default_1, %argmin]), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%index_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, 4), kwargs = {})
triton_poi_fused_div_index_mean_sum_2 = async_compile.triton('triton_poi_fused_div_index_mean_sum_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_index_mean_sum_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_index_mean_sum_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (1))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (2))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp46 = tl.load(in_ptr0 + (3))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp2 = tl.full([XBLOCK], 96, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 96), "index out of bounds: 0 <= tmp5 < 96")
    tmp7 = tl.load(in_ptr1 + (tmp5), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (96 + tmp5), None, eviction_policy='evict_last')
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr1 + (192 + tmp5), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr1 + (288 + tmp5), None, eviction_policy='evict_last')
    tmp13 = tmp11 + tmp12
    tmp14 = 4.0
    tmp15 = tmp13 / tmp14
    tmp18 = tmp17 + tmp2
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert((0 <= tmp20) & (tmp20 < 96), "index out of bounds: 0 <= tmp20 < 96")
    tmp22 = tl.load(in_ptr1 + (384 + tmp20), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (480 + tmp20), None, eviction_policy='evict_last')
    tmp24 = tmp22 + tmp23
    tmp25 = tl.load(in_ptr1 + (576 + tmp20), None, eviction_policy='evict_last')
    tmp26 = tmp24 + tmp25
    tmp27 = tl.load(in_ptr1 + (672 + tmp20), None, eviction_policy='evict_last')
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28 / tmp14
    tmp30 = tmp15 + tmp29
    tmp33 = tmp32 + tmp2
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 96), "index out of bounds: 0 <= tmp35 < 96")
    tmp37 = tl.load(in_ptr1 + (768 + tmp35), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr1 + (864 + tmp35), None, eviction_policy='evict_last')
    tmp39 = tmp37 + tmp38
    tmp40 = tl.load(in_ptr1 + (960 + tmp35), None, eviction_policy='evict_last')
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr1 + (1056 + tmp35), None, eviction_policy='evict_last')
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43 / tmp14
    tmp45 = tmp30 + tmp44
    tmp48 = tmp47 + tmp2
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tl.device_assert((0 <= tmp50) & (tmp50 < 96), "index out of bounds: 0 <= tmp50 < 96")
    tmp52 = tl.load(in_ptr1 + (1152 + tmp50), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr1 + (1248 + tmp50), None, eviction_policy='evict_last')
    tmp54 = tmp52 + tmp53
    tmp55 = tl.load(in_ptr1 + (1344 + tmp50), None, eviction_policy='evict_last')
    tmp56 = tmp54 + tmp55
    tmp57 = tl.load(in_ptr1 + (1440 + tmp50), None, eviction_policy='evict_last')
    tmp58 = tmp56 + tmp57
    tmp59 = tmp58 / tmp14
    tmp60 = tmp45 + tmp59
    tmp61 = 0.25
    tmp62 = tmp60 * tmp61
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp62, None)
''', device_str='cuda')


# kernel path: inductor_cache/iw/ciwgm4jhmy6o77amwuwr7tyiwe7lakb2yoscfjfb4vemsb4b3nt4.py
# Topologically Sorted Source Nodes: [perm_labels], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   perm_labels => index_2
# Graph fragment:
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view, [%iota_default, None, %argmin]), kwargs = {})
triton_poi_fused_index_3 = async_compile.triton('triton_poi_fused_index_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x0 = (xindex % 4)
    x3 = xindex // 4
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 96, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 96)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 96")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4 + 384*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (96, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [getitem], Original ATen: [aten.index]
        buf0 = torch.ops.aten.index.Tensor(arg1_1, [None, None, arg2_1])
        del arg1_1
        del arg2_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((4, 4, 96), (384, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [binary_cross_entropy_with_logits, mean], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_binary_cross_entropy_with_logits_mean_0.run(buf1, arg0_1, buf2, 1536, grid=grid(1536), stream=stream0)
        del arg0_1
        buf3 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [loss_all_case, min_idx], Original ATen: [aten.mean, aten.argmin]
        stream0 = get_raw_stream(0)
        triton_per_fused_argmin_mean_1.run(buf2, buf3, 4, 96, grid=grid(4), stream=stream0)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [loss_all_case, selected_loss, sum_1, pit_min_loss], Original ATen: [aten.mean, aten.index, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_index_mean_sum_2.run(buf6, buf3, buf2, 1, grid=grid(1), stream=stream0)
        del buf2
        buf5 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [perm_labels], Original ATen: [aten.index]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_3.run(buf3, buf1, buf5, 64, grid=grid(64), stream=stream0)
        del buf1
        del buf3
    return (buf6, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
