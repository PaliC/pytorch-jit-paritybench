# AOT ID: ['0_inference']
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


# kernel path: inductor_cache/3j/c3jgqq5nhr4ewxgx7picbkqyw4443ggt65m5b72bsnpm2gne3w5q.py
# Topologically Sorted Source Nodes: [neg, sub, mul, neg_1, tanh, sub_1, abs_1, clamp, log, f1, mul_2, tanh_2, c1, mul_6, neg_4, add_5, pow_1, neg_5, sub_2, pow_2, mul_3, add_6, truediv, f1_sq, neg_8, mul_5, tanh_3, c2, mul_7, f1_1], Original ATen: [aten.neg, aten.sub, aten.mul, aten.tanh, aten.abs, aten.clamp, aten.log, aten.add, aten.pow, aten.div]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   add_5 => add_5
#   add_6 => add_6
#   c1 => clamp_min_2
#   c2 => clamp_min_3
#   clamp => clamp_min
#   f1 => add
#   f1_1 => add_8
#   f1_sq => sub_3
#   log => log
#   mul => mul
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   neg => neg
#   neg_1 => neg_1
#   neg_4 => neg_4
#   neg_5 => neg_5
#   neg_8 => neg_8
#   pow_1 => pow_1
#   pow_2 => pow_2
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   tanh => tanh
#   tanh_2 => tanh_2
#   tanh_3 => tanh_3
#   truediv => div
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, 7), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 0.5), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select_1,), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%neg_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %tanh), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_1, 5e-06), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%clamp_min,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, 6), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, 0.5), kwargs = {})
#   %tanh_2 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_2,), kwargs = {})
#   %clamp_min_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%tanh_2, 0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %clamp_min_2), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_4, 7), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %neg_5 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg_5, 8), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, 0.1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %mul_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_6, 10), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, 20), kwargs = {})
#   %neg_8 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select_1,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_8, 0.5), kwargs = {})
#   %tanh_3 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_5,), kwargs = {})
#   %clamp_min_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%tanh_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_min_3), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_0 = async_compile.triton('triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = -tmp1
    tmp3 = 7.0
    tmp4 = tmp2 - tmp3
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp9 = -tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = tmp6 - tmp10
    tmp12 = tl_math.abs(tmp11)
    tmp13 = 5e-06
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tl_math.log(tmp14)
    tmp16 = 6.0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp8 * tmp5
    tmp19 = libdevice.tanh(tmp18)
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp2 + tmp3
    tmp24 = tmp23 * tmp23
    tmp25 = 8.0
    tmp26 = tmp9 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = 0.1
    tmp29 = tmp27 * tmp28
    tmp30 = tmp24 + tmp29
    tmp31 = tmp30 * tmp28
    tmp32 = 20.0
    tmp33 = tmp31 - tmp32
    tmp34 = tmp9 * tmp5
    tmp35 = libdevice.tanh(tmp34)
    tmp36 = triton_helpers.maximum(tmp35, tmp20)
    tmp37 = tmp33 * tmp36
    tmp38 = tmp22 + tmp37
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/pv/cpvg7ia6w2vmzwwajdkamws5aq46ltev6bqvbo32d5knajuaa3yg.py
# Topologically Sorted Source Nodes: [mul_2, tanh_2, c1, neg_8, mul_5, tanh_3, c2, neg_2, add_1, mul_1, neg_3, tanh_1, add_2, add_3, abs_2, clamp_1, log_1, f2, mul_8, neg_6, sub_4, pow_3, neg_7, sub_5, pow_4, mul_4, add_7, truediv_1, f2_sq, mul_9, f2_1], Original ATen: [aten.mul, aten.tanh, aten.clamp, aten.neg, aten.add, aten.abs, aten.log, aten.sub, aten.pow, aten.div]
# Source node to ATen node mapping:
#   abs_2 => abs_2
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_7 => add_7
#   c1 => clamp_min_2
#   c2 => clamp_min_3
#   clamp_1 => clamp_min_1
#   f2 => add_4
#   f2_1 => add_9
#   f2_sq => sub_6
#   log_1 => log_1
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_8 => mul_8
#   mul_9 => mul_9
#   neg_2 => neg_2
#   neg_3 => neg_3
#   neg_6 => neg_6
#   neg_7 => neg_7
#   neg_8 => neg_8
#   pow_3 => pow_3
#   pow_4 => pow_4
#   sub_4 => sub_4
#   sub_5 => sub_5
#   tanh_1 => tanh_1
#   tanh_2 => tanh_2
#   tanh_3 => tanh_3
#   truediv_1 => div_1
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, 0.5), kwargs = {})
#   %tanh_2 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_2,), kwargs = {})
#   %clamp_min_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%tanh_2, 0), kwargs = {})
#   %neg_8 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select_1,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_8, 0.5), kwargs = {})
#   %tanh_3 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_5,), kwargs = {})
#   %clamp_min_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%tanh_3, 0), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_2, 3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select_1,), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%neg_3,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %tanh_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 2), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_3,), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_2, 5e-06), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%clamp_min_1,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_1, 6), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %clamp_min_2), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg_6, 7), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %neg_7 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%select_1,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg_7, 8), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, 0.1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_3, %mul_4), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_7, 10), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, 20), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_min_3), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_1 = async_compile.triton('triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = -tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp9 = -tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = tmp6 + tmp10
    tmp12 = 2.0
    tmp13 = tmp11 + tmp12
    tmp14 = tl_math.abs(tmp13)
    tmp15 = 5e-06
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tl_math.log(tmp16)
    tmp18 = 6.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp8 * tmp5
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = 0.0
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = tmp19 * tmp23
    tmp25 = 7.0
    tmp26 = tmp2 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = 8.0
    tmp29 = tmp9 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = 0.1
    tmp32 = tmp30 * tmp31
    tmp33 = tmp27 + tmp32
    tmp34 = tmp33 * tmp31
    tmp35 = 20.0
    tmp36 = tmp34 - tmp35
    tmp37 = tmp9 * tmp5
    tmp38 = libdevice.tanh(tmp37)
    tmp39 = triton_helpers.maximum(tmp38, tmp22)
    tmp40 = tmp36 * tmp39
    tmp41 = tmp24 + tmp40
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp41, None)
''', device_str='cuda')


cpp_fused_stack_2 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                out_ptr0[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr1[static_cast<int64_t>(0L)];
                out_ptr1[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [neg, sub, mul, neg_1, tanh, sub_1, abs_1, clamp, log, f1, mul_2, tanh_2, c1, mul_6, neg_4, add_5, pow_1, neg_5, sub_2, pow_2, mul_3, add_6, truediv, f1_sq, neg_8, mul_5, tanh_3, c2, mul_7, f1_1], Original ATen: [aten.neg, aten.sub, aten.mul, aten.tanh, aten.abs, aten.clamp, aten.log, aten.add, aten.pow, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_0.run(arg0_1, buf0, 1, grid=grid(1), stream=stream0)
    buf1 = empty_strided_cpu((), (), torch.float32)
    buf1.copy_(buf0, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_2, tanh_2, c1, neg_8, mul_5, tanh_3, c2, neg_2, add_1, mul_1, neg_3, tanh_1, add_2, add_3, abs_2, clamp_1, log_1, f2, mul_8, neg_6, sub_4, pow_3, neg_7, sub_5, pow_4, mul_4, add_7, truediv_1, f2_sq, mul_9, f2_1], Original ATen: [aten.mul, aten.tanh, aten.clamp, aten.neg, aten.add, aten.abs, aten.log, aten.sub, aten.pow, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_div_log_mul_neg_pow_sub_tanh_1.run(arg0_1, buf2, 1, grid=grid(1), stream=stream0)
        del arg0_1
    buf3 = empty_strided_cpu((), (), torch.float32)
    buf3.copy_(buf2, False)
    del buf2
    buf6 = empty_strided_cpu((2, ), (1, ), torch.float32)
    buf4 = reinterpret_tensor(buf6, (1, ), (1, ), 0)  # alias
    buf5 = reinterpret_tensor(buf6, (1, ), (1, ), 1)  # alias
    cpp_fused_stack_2(buf1, buf3, buf4, buf5)
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
