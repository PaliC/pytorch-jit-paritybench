# AOT ID: ['12_inference']
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


# kernel path: inductor_cache/za/czao22npzefvgtzaq7ojloafhmkljodyyneomjlyus4qmzsmhi4a.py
# Topologically Sorted Source Nodes: [mul, attn_scores, leaky_relu, attn_scores_1], Original ATen: [aten.mul, aten.sum, aten.leaky_relu, aten._softmax]
# Source node to ATen node mapping:
#   attn_scores => sum_1
#   attn_scores_1 => amax
#   leaky_relu => gt, mul_1, where
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, 4), kwargs = {})
#   %sum_1 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1], True), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sum_1, %mul_1), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [0], True), kwargs = {})
triton_poi_fused__softmax_leaky_relu_mul_sum_0 = async_compile.triton('triton_poi_fused__softmax_leaky_relu_mul_sum_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_leaky_relu_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_leaky_relu_mul_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (64 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (65 + 4*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (66 + 4*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (67 + 4*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (128 + 4*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (129 + 4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (130 + 4*x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (131 + 4*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (192 + 4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (193 + 4*x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr0 + (194 + 4*x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (195 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 4.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp1
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp1
    tmp11 = tmp8 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp18 = tmp17 * tmp1
    tmp20 = tmp19 * tmp1
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp1
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp1
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 > tmp12
    tmp29 = tmp27 * tmp14
    tmp30 = tl.where(tmp28, tmp27, tmp29)
    tmp31 = triton_helpers.maximum(tmp16, tmp30)
    tmp33 = tmp32 * tmp1
    tmp35 = tmp34 * tmp1
    tmp36 = tmp33 + tmp35
    tmp38 = tmp37 * tmp1
    tmp39 = tmp36 + tmp38
    tmp41 = tmp40 * tmp1
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42 > tmp12
    tmp44 = tmp42 * tmp14
    tmp45 = tl.where(tmp43, tmp42, tmp44)
    tmp46 = triton_helpers.maximum(tmp31, tmp45)
    tmp48 = tmp47 * tmp1
    tmp50 = tmp49 * tmp1
    tmp51 = tmp48 + tmp50
    tmp53 = tmp52 * tmp1
    tmp54 = tmp51 + tmp53
    tmp56 = tmp55 * tmp1
    tmp57 = tmp54 + tmp56
    tmp58 = tmp57 > tmp12
    tmp59 = tmp57 * tmp14
    tmp60 = tl.where(tmp58, tmp57, tmp59)
    tmp61 = triton_helpers.maximum(tmp46, tmp60)
    tl.store(out_ptr0 + (x0), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lh/clh2exm3aiqzobzf7bda6deb4hi6j22xovipq3zs7p3egk6x4jvx.py
# Topologically Sorted Source Nodes: [mul, attn_scores, leaky_relu, attn_scores_1], Original ATen: [aten.mul, aten.sum, aten.leaky_relu, aten._softmax]
# Source node to ATen node mapping:
#   attn_scores => sum_1
#   attn_scores_1 => exp, sub
#   leaky_relu => gt, mul_1, where
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, 4), kwargs = {})
#   %sum_1 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1], True), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sum_1, %mul_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_leaky_relu_mul_sum_1 = async_compile.triton('triton_poi_fused__softmax_leaky_relu_mul_sum_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_leaky_relu_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_leaky_relu_mul_sum_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 4.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp1
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp1
    tmp11 = tmp8 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp18 = tmp16 - tmp17
    tmp19 = tl_math.exp(tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vg/cvgmwlnctxtmg4unhajwshaycvphf6pzs2wec7nefiy53cg2s5su.py
# Topologically Sorted Source Nodes: [attn_scores_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_scores_1 => div, sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [0], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_2), kwargs = {})
triton_poi_fused__softmax_2 = async_compile.triton('triton_poi_fused__softmax_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (48 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ni/cnin35f2p3kmh3glf6772cp2cdgbtmfu4qzhr625vkvzab6zp2rq.py
# Topologically Sorted Source Nodes: [attn_scores_1, mul_1, out], Original ATen: [aten._softmax, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   attn_scores_1 => div, sum_2
#   mul_1 => mul_2
#   out => sum_3
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [0], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [0]), kwargs = {})
triton_poi_fused__softmax_mul_sum_3 = async_compile.triton('triton_poi_fused__softmax_mul_sum_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_mul_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_mul_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (16 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (64 + x2), xmask)
    tmp7 = tl.load(in_ptr0 + (32 + x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (128 + x2), xmask)
    tmp11 = tl.load(in_ptr0 + (48 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (192 + x2), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [mul, attn_scores, leaky_relu, attn_scores_1], Original ATen: [aten.mul, aten.sum, aten.leaky_relu, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_leaky_relu_mul_sum_0.run(arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [mul, attn_scores, leaky_relu, attn_scores_1], Original ATen: [aten.mul, aten.sum, aten.leaky_relu, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_leaky_relu_mul_sum_1.run(arg0_1, buf0, buf1, 64, grid=grid(64), stream=stream0)
        del buf0
        buf2 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [attn_scores_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf1, buf2, 64, grid=grid(64), stream=stream0)
        buf3 = reinterpret_tensor(buf1, (4, 4, 4), (16, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_1, mul_1, out], Original ATen: [aten._softmax, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_mul_sum_3.run(buf2, arg0_1, buf3, 64, grid=grid(64), stream=stream0)
        del arg0_1
        del buf2
    return (reinterpret_tensor(buf3, (4, 16), (16, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
