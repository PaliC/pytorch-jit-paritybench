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


# kernel path: inductor_cache/z2/cz2ln3hq3tlz34bqgu22tuqa6dp4bwedne32we76oebob6cl465u.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_relu_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kr/ckrybt22e7iq4gqwsmsgj33s3ed3kpcduupliq6i3rtzxqkc6gew.py
# Topologically Sorted Source Nodes: [log_std_1, std, mul, a, var, log_scale, sub, pow_2, neg, mul_1, truediv, sub_1, sub_2, log_pi, wrapped_log, sub_3, mul_2, softplus, sub_4, mul_3, sum_2, log_pi_1], Original ATen: [aten.clamp, aten.exp, aten.mul, aten.add, aten.pow, aten.log, aten.sub, aten.neg, aten.div, aten.sum, aten.softplus]
# Source node to ATen node mapping:
#   a => add
#   log_pi => sum_1
#   log_pi_1 => sub_5
#   log_scale => log
#   log_std_1 => clamp_max, clamp_min
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   neg => neg
#   pow_2 => pow_2
#   softplus => exp_1, gt, log1p, where
#   std => exp
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sum_2 => sum_2
#   truediv => div
#   var => pow_1
#   wrapped_log => full_default
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%view_7, -20), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 2), kwargs = {})
#   %exp : [num_users=3] = call_function[target=torch.ops.aten.exp.default](args = (%clamp_max,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%normal_functional, %exp), kwargs = {})
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %mul), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%exp, 2), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%exp,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %view_5), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%pow_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 2), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg, %mul_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %log), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, 0.9189385332046727), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sub_2, [1], True), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.6931471805599453), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %add), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, -2), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_2, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul_2, %log1p), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %where), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [1], True), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_1, %sum_2), kwargs = {})
triton_poi_fused_add_clamp_div_exp_log_mul_neg_pow_softplus_sub_sum_1 = async_compile.triton('triton_poi_fused_add_clamp_div_exp_log_mul_neg_pow_softplus_sub_sum_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_exp_log_mul_neg_pow_softplus_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_exp_log_mul_neg_pow_softplus_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp2 = tl.load(in_ptr2 + (x0 + 64*x1), xmask)
    tmp20 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp21 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp22 = tl.load(in_ptr2 + (16 + x0 + 64*x1), xmask)
    tmp38 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp39 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp40 = tl.load(in_ptr2 + (32 + x0 + 64*x1), xmask)
    tmp56 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp57 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp58 = tl.load(in_ptr2 + (48 + x0 + 64*x1), xmask)
    tmp3 = -20.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 2.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp0 + tmp8
    tmp10 = tmp9 - tmp0
    tmp11 = tmp10 * tmp10
    tmp12 = -tmp11
    tmp13 = tmp7 * tmp7
    tmp14 = tmp13 * tmp5
    tmp15 = tmp12 / tmp14
    tmp16 = tl_math.log(tmp7)
    tmp17 = tmp15 - tmp16
    tmp18 = 0.9189385332046727
    tmp19 = tmp17 - tmp18
    tmp23 = triton_helpers.maximum(tmp22, tmp3)
    tmp24 = triton_helpers.minimum(tmp23, tmp5)
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tmp21 * tmp25
    tmp27 = tmp20 + tmp26
    tmp28 = tmp27 - tmp20
    tmp29 = tmp28 * tmp28
    tmp30 = -tmp29
    tmp31 = tmp25 * tmp25
    tmp32 = tmp31 * tmp5
    tmp33 = tmp30 / tmp32
    tmp34 = tl_math.log(tmp25)
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 - tmp18
    tmp37 = tmp19 + tmp36
    tmp41 = triton_helpers.maximum(tmp40, tmp3)
    tmp42 = triton_helpers.minimum(tmp41, tmp5)
    tmp43 = tl_math.exp(tmp42)
    tmp44 = tmp39 * tmp43
    tmp45 = tmp38 + tmp44
    tmp46 = tmp45 - tmp38
    tmp47 = tmp46 * tmp46
    tmp48 = -tmp47
    tmp49 = tmp43 * tmp43
    tmp50 = tmp49 * tmp5
    tmp51 = tmp48 / tmp50
    tmp52 = tl_math.log(tmp43)
    tmp53 = tmp51 - tmp52
    tmp54 = tmp53 - tmp18
    tmp55 = tmp37 + tmp54
    tmp59 = triton_helpers.maximum(tmp58, tmp3)
    tmp60 = triton_helpers.minimum(tmp59, tmp5)
    tmp61 = tl_math.exp(tmp60)
    tmp62 = tmp57 * tmp61
    tmp63 = tmp56 + tmp62
    tmp64 = tmp63 - tmp56
    tmp65 = tmp64 * tmp64
    tmp66 = -tmp65
    tmp67 = tmp61 * tmp61
    tmp68 = tmp67 * tmp5
    tmp69 = tmp66 / tmp68
    tmp70 = tl_math.log(tmp61)
    tmp71 = tmp69 - tmp70
    tmp72 = tmp71 - tmp18
    tmp73 = tmp55 + tmp72
    tmp74 = 0.6931471805599453
    tmp75 = tmp74 - tmp9
    tmp76 = -2.0
    tmp77 = tmp9 * tmp76
    tmp78 = 20.0
    tmp79 = tmp77 > tmp78
    tmp80 = tl_math.exp(tmp77)
    tmp81 = libdevice.log1p(tmp80)
    tmp82 = tl.where(tmp79, tmp77, tmp81)
    tmp83 = tmp75 - tmp82
    tmp84 = tmp83 * tmp5
    tmp85 = tmp74 - tmp27
    tmp86 = tmp27 * tmp76
    tmp87 = tmp86 > tmp78
    tmp88 = tl_math.exp(tmp86)
    tmp89 = libdevice.log1p(tmp88)
    tmp90 = tl.where(tmp87, tmp86, tmp89)
    tmp91 = tmp85 - tmp90
    tmp92 = tmp91 * tmp5
    tmp93 = tmp84 + tmp92
    tmp94 = tmp74 - tmp45
    tmp95 = tmp45 * tmp76
    tmp96 = tmp95 > tmp78
    tmp97 = tl_math.exp(tmp95)
    tmp98 = libdevice.log1p(tmp97)
    tmp99 = tl.where(tmp96, tmp95, tmp98)
    tmp100 = tmp94 - tmp99
    tmp101 = tmp100 * tmp5
    tmp102 = tmp93 + tmp101
    tmp103 = tmp74 - tmp63
    tmp104 = tmp63 * tmp76
    tmp105 = tmp104 > tmp78
    tmp106 = tl_math.exp(tmp104)
    tmp107 = libdevice.log1p(tmp106)
    tmp108 = tl.where(tmp105, tmp104, tmp107)
    tmp109 = tmp103 - tmp108
    tmp110 = tmp109 * tmp5
    tmp111 = tmp102 + tmp110
    tmp112 = tmp73 - tmp111
    tl.store(in_out_ptr0 + (x2), tmp112, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/at/catf6sux7le2fgbne2mx42e23zjubyabnxx72a3roax3lbmd7cax.py
# Topologically Sorted Source Nodes: [log_std_1, std, mul, a, tanh, a_1], Original ATen: [aten.clamp, aten.exp, aten.mul, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   a => add
#   a_1 => mul_4
#   log_std_1 => clamp_max, clamp_min
#   mul => mul
#   std => exp
#   tanh => tanh
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%view_7, -20), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 2), kwargs = {})
#   %exp : [num_users=3] = call_function[target=torch.ops.aten.exp.default](args = (%clamp_max,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%normal_functional, %exp), kwargs = {})
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %mul), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 4), kwargs = {})
triton_poi_fused_add_clamp_exp_mul_tanh_2 = async_compile.triton('triton_poi_fused_add_clamp_exp_mul_tanh_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_mul_tanh_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_mul_tanh_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = -20.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 2.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp0 + tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = 4.0
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4), (4, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf0  # reuse
        buf14 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf1, primals_2, buf14, 256, grid=grid(256), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf2)
        buf3 = reinterpret_tensor(buf2, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf2  # reuse
        buf13 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf3, primals_5, buf13, 256, grid=grid(256), stream=stream0)
        del primals_5
        buf4 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, reinterpret_tensor(buf3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf4)
        del primals_7
        buf5 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_std], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_8, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf5)
        del primals_9
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eps], Original ATen: [aten.normal_functional]
        buf7 = torch.ops.aten.normal_functional.default(buf6)
        buf8 = buf7
        del buf7
        buf9 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf11 = reinterpret_tensor(buf9, (4, 1, 4, 4), (16, 16, 4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [log_std_1, std, mul, a, var, log_scale, sub, pow_2, neg, mul_1, truediv, sub_1, sub_2, log_pi, wrapped_log, sub_3, mul_2, softplus, sub_4, mul_3, sum_2, log_pi_1], Original ATen: [aten.clamp, aten.exp, aten.mul, aten.add, aten.pow, aten.log, aten.sub, aten.neg, aten.div, aten.sum, aten.softplus]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_exp_log_mul_neg_pow_softplus_sub_sum_1.run(buf11, buf4, buf8, buf5, 64, grid=grid(64), stream=stream0)
        buf12 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [log_std_1, std, mul, a, tanh, a_1], Original ATen: [aten.clamp, aten.exp, aten.mul, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_exp_mul_tanh_2.run(buf4, buf8, buf5, buf12, 256, grid=grid(256), stream=stream0)
    return (buf12, buf11, reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(buf1, (64, 4), (4, 1), 0), reinterpret_tensor(buf3, (64, 4), (4, 1), 0), buf4, buf5, buf8, primals_8, primals_6, buf13, primals_4, buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
