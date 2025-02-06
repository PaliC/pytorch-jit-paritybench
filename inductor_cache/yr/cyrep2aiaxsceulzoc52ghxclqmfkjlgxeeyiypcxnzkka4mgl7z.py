# AOT ID: ['18_inference']
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


# kernel path: inductor_cache/sf/csf4rskm2qgfbpys57akmhok4dcoznaksrdkj23vsyb4m6bdeif2.py
# Topologically Sorted Source Nodes: [n, log, mul, k, log_1, mul_1, sub_1, sub_2, sub_3, add_2, log_2, mul_2, sub_4, x, log_3, mul_3, add_3, sub_5, sub, one_minus_x, log_4, mul_4, y], Original ATen: [aten.add, aten.log, aten.mul, aten.sub, aten.clamp, aten.rsub]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   k => add_1
#   log => log
#   log_1 => log_1
#   log_2 => log_2
#   log_3 => log_3
#   log_4 => log_4
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   n => add
#   one_minus_x => clamp_max, clamp_min
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   x => clamp_max_1, clamp_min_1
#   y => add_4
# Graph fragment:
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, 1e-07), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %log), kwargs = {})
#   %add_1 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, 1e-07), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %log_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %add_1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %add_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_3, 1e-07), kwargs = {})
#   %log_2 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %log_2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %mul_2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg0_1, 0.0001), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1), kwargs = {})
#   %log_3 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%clamp_max_1,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %log_3), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_4, %mul_3), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (255, %arg2_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0001), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
#   %log_4 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%clamp_max,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %log_4), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_4), kwargs = {})
triton_poi_fused_add_clamp_log_mul_rsub_sub_0 = async_compile.triton('triton_poi_fused_add_clamp_log_mul_rsub_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_log_mul_rsub_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_log_mul_rsub_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (x3), None)
    tmp2 = 1e-07
    tmp3 = tmp1 + tmp2
    tmp4 = tl_math.log(tmp3)
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 + tmp2
    tmp9 = tl_math.log(tmp8)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 - tmp10
    tmp12 = tmp3 - tmp8
    tmp13 = tmp12 + tmp2
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp12 * tmp14
    tmp16 = tmp11 - tmp15
    tmp18 = 0.0001
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 1.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tl_math.log(tmp21)
    tmp23 = tmp7 * tmp22
    tmp24 = tmp16 + tmp23
    tmp25 = tl.full([1], 255, tl.int64)
    tmp26 = tmp25 - tmp6
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp20 - tmp17
    tmp29 = triton_helpers.maximum(tmp28, tmp18)
    tmp30 = triton_helpers.minimum(tmp29, tmp20)
    tmp31 = tl_math.log(tmp30)
    tmp32 = tmp27 * tmp31
    tmp33 = tmp24 + tmp32
    tl.store(out_ptr0 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/gj/cgj7cqkvy23sxxosrhu77wgjsksyxohjjoeyciaqfld5eocqp3p4.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 16)
    x1 = xindex // 16
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 2048*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uo/cuo4fy6vrtejc5mi3vdwz5vrcophl4toobcvvntjpaf7vg5nrqz7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
triton_per_fused_2 = async_compile.triton('triton_per_fused_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 32*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yk/cyk3jgu7r7hosikwsvffmwl5zuyrvgcu3hhhcfybzktmdxeleqgg.py
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => exp, sum_1
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__softmax_3 = async_compile.triton('triton_red_fused__softmax_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_3(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 16)
    x4 = xindex // 16
    x2 = xindex // 32
    tmp3 = tl.load(in_ptr1 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r3 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 - tmp3
        tmp5 = tmp4 * tmp1
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pd/cpdjedp3a2xuu26cvm5itknkuxljgqjsf7mbshjiask7oimdxxuu.py
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => exp, sum_1
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_per_fused__softmax_4 = async_compile.triton('triton_per_fused__softmax_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 32*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nz/cnzvzrczioeczbrnqb74rwh5sgxu4abqiu6pe5xfbxcn7gfj5ylu.py
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => div_1, exp
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_5 = async_compile.triton('triton_poi_fused__softmax_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 16*x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x0 + 16*x2), None, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tmp4 * tmp1
    tmp6 = tl_math.exp(tmp5)
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(arg1_1, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg2_1, (1, 256, 1, 1), (256, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [n, log, mul, k, log_1, mul_1, sub_1, sub_2, sub_3, add_2, log_2, mul_2, sub_4, x, log_3, mul_3, add_3, sub_5, sub, one_minus_x, log_4, mul_4, y], Original ATen: [aten.add, aten.log, aten.mul, aten.sub, aten.clamp, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_log_mul_rsub_sub_0.run(arg1_1, arg2_1, arg0_1, buf0, 16384, grid=grid(16384), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        buf1 = empty_strided_cuda((4, 1, 4, 4, 2), (32, 128, 4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_red_fused_1.run(buf0, buf1, 128, 128, grid=grid(128), stream=stream0)
        buf2 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_per_fused_2.run(buf1, buf2, 64, 2, grid=grid(64), stream=stream0)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_3.run(buf0, buf2, buf3, 128, 128, grid=grid(128), stream=stream0)
        buf4 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_4.run(buf3, buf4, 64, 2, grid=grid(64), stream=stream0)
        del buf3
        buf5 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf5, buf2, buf4, 16384, grid=grid(16384), stream=stream0)
        del buf2
        del buf4
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
