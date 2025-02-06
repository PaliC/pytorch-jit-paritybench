# AOT ID: ['26_inference']
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


# kernel path: inductor_cache/6c/c6crry7wk3tms5k7c6r2q7ldtfepc3mozje2aitu53pl7564wk5p.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   target => exp
# Graph fragment:
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 10), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
triton_poi_fused__softmax_0 = async_compile.triton('triton_poi_fused__softmax_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 0.1
    tmp16 = tmp14 * tmp15
    tmp17 = tl_math.exp(tmp16)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wx/cwxcrbnklwz6w53ralqr4hsjdu4fcj5udnv533dnsk72zb62kmtp.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   target => div_1, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div_1 : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_1 = async_compile.triton('triton_poi_fused__softmax_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xi/cxif6opsrjwx7e4m3n6xosb76aqowyxwij5s46xobyqhsbinyfnm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 10), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 0.1
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/td/ctdfapw2izr2mkumenixxamg3pcgsafaguaojlt4dhnp36yb77cd.py
# Topologically Sorted Source Nodes: [kl_div, log_softmax, mean, kd_loss, loss, loss_kd], Original ATen: [aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   kd_loss => mul_2
#   kl_div => eq, full_default, full_default_1, isnan, log_1, mul, mul_1, sub_3, where, where_1
#   log_softmax => exp_1, log, sub_2, sum_2
#   loss => mean_1
#   loss_kd => mul_3
#   mean => mean
# Graph fragment:
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%div_1,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%div_1, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %log_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul_1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_1, %where), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor, %log), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_3, [1]), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 100), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 1.0), kwargs = {})
triton_per_fused__log_softmax_mean_mul_sub_xlogy_3 = async_compile.triton('triton_per_fused__log_softmax_mean_mul_sub_xlogy_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_mean_mul_sub_xlogy_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_mean_mul_sub_xlogy_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp11 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp17 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp24 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp35 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp46 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp1 = libdevice.isnan(tmp0).to(tl.int1)
    tmp2 = 0.0
    tmp3 = tmp0 == tmp2
    tmp4 = tl_math.log(tmp0)
    tmp5 = tmp0 * tmp4
    tmp6 = tl.where(tmp3, tmp2, tmp5)
    tmp7 = float("nan")
    tmp8 = tl.where(tmp1, tmp7, tmp6)
    tmp10 = tl_math.exp(tmp9)
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tmp10 + tmp12
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp13 + tmp15
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp20 = tl_math.log(tmp19)
    tmp21 = tmp9 - tmp20
    tmp22 = tmp0 * tmp21
    tmp23 = tmp8 - tmp22
    tmp25 = libdevice.isnan(tmp24).to(tl.int1)
    tmp26 = tmp24 == tmp2
    tmp27 = tl_math.log(tmp24)
    tmp28 = tmp24 * tmp27
    tmp29 = tl.where(tmp26, tmp2, tmp28)
    tmp30 = tl.where(tmp25, tmp7, tmp29)
    tmp31 = tmp11 - tmp20
    tmp32 = tmp24 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tmp23 + tmp33
    tmp36 = libdevice.isnan(tmp35).to(tl.int1)
    tmp37 = tmp35 == tmp2
    tmp38 = tl_math.log(tmp35)
    tmp39 = tmp35 * tmp38
    tmp40 = tl.where(tmp37, tmp2, tmp39)
    tmp41 = tl.where(tmp36, tmp7, tmp40)
    tmp42 = tmp14 - tmp20
    tmp43 = tmp35 * tmp42
    tmp44 = tmp41 - tmp43
    tmp45 = tmp34 + tmp44
    tmp47 = libdevice.isnan(tmp46).to(tl.int1)
    tmp48 = tmp46 == tmp2
    tmp49 = tl_math.log(tmp46)
    tmp50 = tmp46 * tmp49
    tmp51 = tl.where(tmp48, tmp2, tmp50)
    tmp52 = tl.where(tmp47, tmp7, tmp51)
    tmp53 = tmp17 - tmp20
    tmp54 = tmp46 * tmp53
    tmp55 = tmp52 - tmp54
    tmp56 = tmp45 + tmp55
    tmp57 = 4.0
    tmp58 = tmp56 / tmp57
    tmp59 = 100.0
    tmp60 = tmp58 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
    tmp63 = tl.sum(tmp61, 1)[:, None]
    tmp64 = 64.0
    tmp65 = tmp63 / tmp64
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp67, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg1_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf0, buf1, 256, grid=grid(256), stream=stream0)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(arg0_1, buf2, 256, grid=grid(256), stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [kl_div, log_softmax, mean, kd_loss, loss, loss_kd], Original ATen: [aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_mean_mul_sub_xlogy_3.run(buf5, buf1, buf2, 1, 64, grid=grid(1), stream=stream0)
        del buf1
        del buf2
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
