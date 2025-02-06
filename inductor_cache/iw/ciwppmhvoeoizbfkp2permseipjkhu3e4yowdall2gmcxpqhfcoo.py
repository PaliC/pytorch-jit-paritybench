# AOT ID: ['5_inference']
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


# kernel path: inductor_cache/jo/cjo4xvvelfq6ux7ngwj7djne2qktw6blw3jmdl2kpbsyawvxle4m.py
# Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   p_s => exp
# Graph fragment:
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 2), kwargs = {})
#   %mul_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %amax_default_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_2, [-1], True), kwargs = {})
#   %sub_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_2, %amax_default_2), kwargs = {})
#   %div_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_2, 2), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_2,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
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
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tl_math.exp(tmp16)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ry/cryk7ey74danxp6t4ch4lfscj5ml6xl53kb2hmfcfcotgflxrgeo.py
# Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   p_s => div_1, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
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
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dg/cdgbbizjssuqfhrbjrsof5b2d4n6bn3c7fo3bzmb7wwyaivb5lqa.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 2), kwargs = {})
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
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
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
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zf/czfecyk23e2goqr7kaeaszzrltwruaxzl5bonp6wkfbph5tmq3vv.py
# Topologically Sorted Source Nodes: [p_s1, mul, sum_1, mean, p_t, mul_1, sum_2, mean_1, loss, mul_2], Original ATen: [aten._log_softmax, aten.mul, aten.sum, aten.mean, aten.sub]
# Source node to ATen node mapping:
#   loss => sub_5
#   mean => mean
#   mean_1 => mean_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   p_s1 => exp_1, log, sub_2, sum_2
#   p_t => exp_2, log_1, sub_4, sum_3
#   sum_1 => sum_4
#   sum_2 => sum_5
# Graph fragment:
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor_1, %log), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %div_1), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1]), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_4,), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1], True), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_3,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor, %log_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %div_1), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [-1]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_5,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, 0.1), kwargs = {})
triton_per_fused__log_softmax_mean_mul_sub_sum_3 = async_compile.triton('triton_per_fused__log_softmax_mean_mul_sub_sum_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_mean_mul_sub_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_mean_mul_sub_sum_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp3 = tl_math.exp(tmp2)
    tmp4 = tmp1 + tmp3
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tmp4 + tmp6
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tmp7 + tmp9
    tmp11 = tl_math.log(tmp10)
    tmp12 = tmp0 - tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp2 - tmp11
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp5 - tmp11
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp8 - tmp11
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.sum(tmp27, 1)[:, None]
    tmp31 = tl_math.exp(tmp30)
    tmp33 = tl_math.exp(tmp32)
    tmp34 = tmp31 + tmp33
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tmp34 + tmp36
    tmp39 = tl_math.exp(tmp38)
    tmp40 = tmp37 + tmp39
    tmp41 = tl_math.log(tmp40)
    tmp42 = tmp30 - tmp41
    tmp43 = tmp42 * tmp13
    tmp44 = tmp32 - tmp41
    tmp45 = tmp44 * tmp16
    tmp46 = tmp43 + tmp45
    tmp47 = tmp35 - tmp41
    tmp48 = tmp47 * tmp20
    tmp49 = tmp46 + tmp48
    tmp50 = tmp38 - tmp41
    tmp51 = tmp50 * tmp24
    tmp52 = tmp49 + tmp51
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
    tmp55 = tl.sum(tmp53, 1)[:, None]
    tmp56 = 64.0
    tmp57 = tmp29 / tmp56
    tmp58 = tmp55 / tmp56
    tmp59 = tmp57 - tmp58
    tmp60 = 0.1
    tmp61 = tmp59 * tmp60
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp61, None)
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
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg0_1, buf0, buf1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf1, buf2, 256, grid=grid(256), stream=stream0)
        buf4 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(arg1_1, buf4, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf6 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [p_s1, mul, sum_1, mean, p_t, mul_1, sum_2, mean_1, loss, mul_2], Original ATen: [aten._log_softmax, aten.mul, aten.sum, aten.mean, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_mean_mul_sub_sum_3.run(buf6, buf0, buf2, buf4, 1, 64, grid=grid(1), stream=stream0)
        del buf0
        del buf2
        del buf4
    return (buf6, )


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
