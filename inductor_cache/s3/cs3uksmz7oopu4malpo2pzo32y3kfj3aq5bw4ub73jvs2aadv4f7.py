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


# kernel path: inductor_cache/zp/czp35rtqpmtwk6q4hww5ukuytylpkbmm4232sdx4ja2dsxqlxtvb.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   logits => convolution
# Graph fragment:
#   %convolution : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6y/c6ywxbhpde76tyijneuwqsduok7pgaveok5hspuhaquiajslu73z.py
# Topologically Sorted Source Nodes: [soft_one_hot], Original ATen: [aten.log, aten.neg, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   soft_one_hot => add, exp, log, neg, sum_1
# Graph fragment:
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%exponential,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%log,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %neg), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_poi_fused__softmax_add_log_neg_1 = async_compile.triton('triton_poi_fused__softmax_add_log_neg_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_log_neg_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_log_neg_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp8 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp14 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp21 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp22 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp2 = tl_math.log(tmp1)
    tmp3 = -tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp9 = tl_math.log(tmp8)
    tmp10 = -tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp11 * tmp5
    tmp13 = triton_helpers.maximum(tmp6, tmp12)
    tmp16 = tl_math.log(tmp15)
    tmp17 = -tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp18 * tmp5
    tmp20 = triton_helpers.maximum(tmp13, tmp19)
    tmp23 = tl_math.log(tmp22)
    tmp24 = -tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = tmp25 * tmp5
    tmp27 = triton_helpers.maximum(tmp20, tmp26)
    tmp28 = tmp6 - tmp27
    tmp29 = tmp28 * tmp5
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp12 - tmp27
    tmp32 = tmp31 * tmp5
    tmp33 = tl_math.exp(tmp32)
    tmp34 = tmp30 + tmp33
    tmp35 = tmp19 - tmp27
    tmp36 = tmp35 * tmp5
    tmp37 = tl_math.exp(tmp36)
    tmp38 = tmp34 + tmp37
    tmp39 = tmp26 - tmp27
    tmp40 = tmp39 * tmp5
    tmp41 = tl_math.exp(tmp40)
    tmp42 = tmp38 + tmp41
    tl.store(out_ptr0 + (x2), tmp27, xmask)
    tl.store(out_ptr1 + (x2), tmp42, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2d/c2dygcrgojnyyxerb53iarjyqe3m3qdlc252tfxl3ypg77vltwvx.py
# Topologically Sorted Source Nodes: [soft_one_hot, qy], Original ATen: [aten.log, aten.neg, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   qy => amax_1, exp_1, sub_2
#   soft_one_hot => add, div_1, exp, log, neg
# Graph fragment:
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%exponential,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%log,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %neg), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 1.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %div_1 : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convolution, [1], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
triton_poi_fused__softmax_add_log_neg_2 = async_compile.triton('triton_poi_fused__softmax_add_log_neg_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_log_neg_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_log_neg_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr1 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl_math.log(tmp1)
    tmp3 = -tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp9 = tmp8 * tmp5
    tmp10 = tl_math.exp(tmp9)
    tmp12 = tmp10 / tmp11
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = tmp0 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chwhniticn67alxfq5xma6gcki63q7qacbeojcuwy4xtg6z3urep.py
# Topologically Sorted Source Nodes: [qy], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   qy => div_2, sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_2), kwargs = {})
triton_poi_fused__softmax_3 = async_compile.triton('triton_poi_fused__softmax_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/pi/cpinawmmwno4svyj5bhkno3mi5msui47ncxt7qvslqrcufl4hcuk.py
# Topologically Sorted Source Nodes: [mul, add, log, mul_1, sum_1, mean, diff], Original ATen: [aten.mul, aten.add, aten.log, aten.sum, aten.mean]
# Source node to ATen node mapping:
#   add => add_2
#   diff => mul_2
#   log => log_1
#   mean => mean
#   mul => mul
#   mul_1 => mul_1
#   sum_1 => sum_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 4), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 1e-10), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %log_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 0.0005), kwargs = {})
triton_per_fused_add_log_mean_mul_sum_4 = async_compile.triton('triton_per_fused_add_log_mean_mul_sum_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_log_mean_mul_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_log_mean_mul_sum_4(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp7 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp19 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp1 = 4.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1e-10
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.log(tmp4)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp7 * tmp1
    tmp9 = tmp8 + tmp3
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp14 = tmp13 * tmp1
    tmp15 = tmp14 + tmp3
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp13 * tmp16
    tmp18 = tmp12 + tmp17
    tmp20 = tmp19 * tmp1
    tmp21 = tmp20 + tmp3
    tmp22 = tl_math.log(tmp21)
    tmp23 = tmp19 * tmp22
    tmp24 = tmp18 + tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp28 = 64.0
    tmp29 = tmp27 / tmp28
    tmp30 = 0.0005
    tmp31 = tmp29 * tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/bv/cbvw55ep3iuww7mrywcuqagagkm5sjiinxizq7f23dcinnyr6nhx.py
# Topologically Sorted Source Nodes: [soft_one_hot, ind], Original ATen: [aten.max, aten.scatter, aten.sub, aten.add, aten.argmax]
# Source node to ATen node mapping:
#   ind => argmax
#   soft_one_hot => add_1, max_1, scatter_upon_const_tensor, sub_1
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%div_1, 1, True), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [4, 4, 4, 4], background_val: 0, dtype: torch.float32, dim: 1, selector: %getitem_1, val: 1.0})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%scatter_upon_const_tensor, %div_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, %div_1), kwargs = {})
#   %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%add_1, 1), kwargs = {})
triton_poi_fused_add_argmax_max_scatter_sub_5 = async_compile.triton('triton_poi_fused_add_argmax_max_scatter_sub_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_argmax_max_scatter_sub_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_argmax_max_scatter_sub_5(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp32 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = tmp0 != tmp0
    tmp5 = tmp1 != tmp1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp2 | tmp6
    tmp8 = tmp4 & tmp5
    tmp9 = tmp3 | tmp8
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tmp7 | tmp13
    tmp15 = tl.where(tmp14, tmp0, tmp1)
    tmp16 = tl.where(tmp14, tmp10, tmp11)
    tmp18 = tmp15 > tmp17
    tmp19 = tmp15 == tmp17
    tmp20 = tmp15 != tmp15
    tmp21 = tmp17 != tmp17
    tmp22 = tmp20 > tmp21
    tmp23 = tmp18 | tmp22
    tmp24 = tmp20 & tmp21
    tmp25 = tmp19 | tmp24
    tmp26 = tl.full([1], 2, tl.int64)
    tmp27 = tmp16 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tmp23 | tmp28
    tmp30 = tl.where(tmp29, tmp15, tmp17)
    tmp31 = tl.where(tmp29, tmp16, tmp26)
    tmp33 = tmp30 > tmp32
    tmp34 = tmp30 == tmp32
    tmp35 = tmp30 != tmp30
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tl.full([1], 3, tl.int64)
    tmp42 = tmp31 < tmp41
    tmp43 = tmp40 & tmp42
    tmp44 = tmp38 | tmp43
    tmp45 = tl.where(tmp44, tmp30, tmp32)
    tmp46 = tl.where(tmp44, tmp31, tmp41)
    tmp47 = tmp46 == tmp10
    tmp48 = 1.0
    tmp49 = 0.0
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tmp50 - tmp0
    tmp52 = tmp51 + tmp0
    tmp53 = tmp46 == tmp11
    tmp54 = tl.where(tmp53, tmp48, tmp49)
    tmp55 = tmp54 - tmp1
    tmp56 = tmp55 + tmp1
    tmp57 = tmp52 > tmp56
    tmp58 = tmp52 == tmp56
    tmp59 = tmp52 != tmp52
    tmp60 = tmp56 != tmp56
    tmp61 = tmp59 > tmp60
    tmp62 = tmp57 | tmp61
    tmp63 = tmp59 & tmp60
    tmp64 = tmp58 | tmp63
    tmp65 = tmp64 & tmp12
    tmp66 = tmp62 | tmp65
    tmp67 = tl.where(tmp66, tmp52, tmp56)
    tmp68 = tl.where(tmp66, tmp10, tmp11)
    tmp69 = tmp46 == tmp26
    tmp70 = tl.where(tmp69, tmp48, tmp49)
    tmp71 = tmp70 - tmp17
    tmp72 = tmp71 + tmp17
    tmp73 = tmp67 > tmp72
    tmp74 = tmp67 == tmp72
    tmp75 = tmp67 != tmp67
    tmp76 = tmp72 != tmp72
    tmp77 = tmp75 > tmp76
    tmp78 = tmp73 | tmp77
    tmp79 = tmp75 & tmp76
    tmp80 = tmp74 | tmp79
    tmp81 = tmp68 < tmp26
    tmp82 = tmp80 & tmp81
    tmp83 = tmp78 | tmp82
    tmp84 = tl.where(tmp83, tmp67, tmp72)
    tmp85 = tl.where(tmp83, tmp68, tmp26)
    tmp86 = tmp46 == tmp41
    tmp87 = tl.where(tmp86, tmp48, tmp49)
    tmp88 = tmp87 - tmp32
    tmp89 = tmp88 + tmp32
    tmp90 = tmp84 > tmp89
    tmp91 = tmp84 == tmp89
    tmp92 = tmp84 != tmp84
    tmp93 = tmp89 != tmp89
    tmp94 = tmp92 > tmp93
    tmp95 = tmp90 | tmp94
    tmp96 = tmp92 & tmp93
    tmp97 = tmp91 | tmp96
    tmp98 = tmp85 < tmp41
    tmp99 = tmp97 & tmp98
    tmp100 = tmp95 | tmp99
    tmp101 = tl.where(tmp100, tmp84, tmp89)
    tmp102 = tl.where(tmp100, tmp85, tmp41)
    tl.store(out_ptr0 + (x2), tmp46, xmask)
    tl.store(out_ptr1 + (x2), tmp102, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5x/c5x7czl2lz3egk576cnoppdwzn4ezguq4hvwmjvk66v26jpm6l4j.py
# Topologically Sorted Source Nodes: [z_q], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   z_q => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex // 4
    y0 = (yindex % 4)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2 + 16*y3), xmask & ymask)
    tmp1 = y0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7 + tmp6
    tl.store(out_ptr0 + (y0 + 4*x2 + 64*y1), tmp8, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf1, primals_2, 256, grid=grid(256), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [soft_one_hot], Original ATen: [aten.exponential]
        buf3 = torch.ops.aten.exponential.default(buf2)
        buf4 = buf3
        del buf3
        buf5 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf6 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [soft_one_hot], Original ATen: [aten.log, aten.neg, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_log_neg_1.run(buf1, buf4, buf5, buf6, 64, grid=grid(64), stream=stream0)
        buf7 = buf4; del buf4  # reuse
        buf11 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [soft_one_hot, qy], Original ATen: [aten.log, aten.neg, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_log_neg_2.run(buf7, buf1, buf5, buf6, buf11, 256, grid=grid(256), stream=stream0)
        del buf5
        del buf6
        buf12 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [qy], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf11, buf12, 256, grid=grid(256), stream=stream0)
        buf13 = empty_strided_cuda((), (), torch.float32)
        buf15 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [mul, add, log, mul_1, sum_1, mean, diff], Original ATen: [aten.mul, aten.add, aten.log, aten.sum, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_log_mean_mul_sum_4.run(buf15, buf12, 1, 64, grid=grid(1), stream=stream0)
        buf8 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.int64)
        buf14 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [soft_one_hot, ind], Original ATen: [aten.max, aten.scatter, aten.sub, aten.add, aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_argmax_max_scatter_sub_5.run(buf7, buf8, buf14, 64, grid=grid(64), stream=stream0)
        buf9 = reinterpret_tensor(buf12, (4, 4, 4, 4, 1), (64, 16, 4, 1, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [z_q], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf8, buf7, buf9, 16, 16, grid=grid(16, 16), stream=stream0)
        del buf8
        buf10 = reinterpret_tensor(buf11, (1, 64, 4), (256, 4, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [z_q], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 64, 4), (0, 4, 1), 0), reinterpret_tensor(primals_4, (1, 4, 4), (16, 4, 1), 0), out=buf10)
    return (reinterpret_tensor(buf10, (4, 4, 4, 4), (64, 1, 16, 4), 0), buf15, buf14, primals_1, primals_3, buf1, buf7, reinterpret_tensor(buf9, (1, 4, 64), (256, 1, 4), 0), reinterpret_tensor(primals_4, (1, 4, 4), (16, 1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
