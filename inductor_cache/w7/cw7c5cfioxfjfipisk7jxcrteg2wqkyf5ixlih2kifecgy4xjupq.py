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


# kernel path: inductor_cache/vl/cvlzzht6oafhg2l3kckmpbfmyluvrjxbdzc34y7gptg2sv5vnetv.py
# Topologically Sorted Source Nodes: [softmax, loss_stage1], Original ATen: [aten._softmax, aten._log_softmax]
# Source node to ATen node mapping:
#   loss_stage1 => amax, sub
#   softmax => amax_2, exp_2, sub_4
# Graph fragment:
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%arg1_1, [1], True), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %amax_2), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%arg1_1, [1], True), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %amax), kwargs = {})
triton_poi_fused__log_softmax__softmax_0 = async_compile.triton('triton_poi_fused__log_softmax__softmax_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__softmax_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a5/ca5rgc3ucrfie66m7aojl43yccx2zz2wlqmg342jqtwrn7vo2dit.py
# Topologically Sorted Source Nodes: [softmax, max_1, softmax_1, max_2, loss_stage1, sub, exponential_term_stage1, loss_stage1_1, loss_stage2, sub_1, exponential_term_stage2, loss_stage2_1, loss], Original ATen: [aten._softmax, aten.max, aten._log_softmax, aten.mul, aten.sum, aten.neg, aten.rsub, aten.pow, aten.add]
# Source node to ATen node mapping:
#   exponential_term_stage1 => pow_1
#   exponential_term_stage2 => pow_2
#   loss => add
#   loss_stage1 => exp, log, mul, neg, sub_1, sum_2, sum_3
#   loss_stage1_1 => mul_2
#   loss_stage2 => exp_1, log_1, mul_1, neg_1, sub_3, sum_4, sum_5
#   loss_stage2_1 => mul_3
#   max_1 => max_1
#   max_2 => max_2
#   softmax => div, sum_6
#   softmax_1 => div_1, sum_7
#   sub => sub_5
#   sub_1 => sub_7
# Graph fragment:
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_2, %sum_6), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%div, 1), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_3, %sum_7), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%div_1, 1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %log), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %arg0_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_3,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %getitem), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %pow_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_4,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_2, %log_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %arg0_1), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_5,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %getitem_2), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %pow_2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
triton_poi_fused__log_softmax__softmax_add_max_mul_neg_pow_rsub_sum_1 = async_compile.triton('triton_poi_fused__log_softmax__softmax_add_max_mul_neg_pow_rsub_sum_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__softmax_add_max_mul_neg_pow_rsub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__softmax_add_max_mul_neg_pow_rsub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp2 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp8 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp13 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp16 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp20 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp24 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp28 = tl.load(in_ptr2 + (x0 + 64*x1), xmask)
    tmp29 = tl.load(in_ptr2 + (16 + x0 + 64*x1), xmask)
    tmp31 = tl.load(in_ptr2 + (32 + x0 + 64*x1), xmask)
    tmp33 = tl.load(in_ptr2 + (48 + x0 + 64*x1), xmask)
    tmp46 = tl.load(in_ptr3 + (x0 + 64*x1), xmask)
    tmp48 = tl.load(in_ptr3 + (16 + x0 + 64*x1), xmask)
    tmp51 = tl.load(in_ptr3 + (32 + x0 + 64*x1), xmask)
    tmp54 = tl.load(in_ptr3 + (48 + x0 + 64*x1), xmask)
    tmp70 = tl.load(in_ptr4 + (x0 + 64*x1), xmask)
    tmp71 = tl.load(in_ptr4 + (16 + x0 + 64*x1), xmask)
    tmp73 = tl.load(in_ptr4 + (32 + x0 + 64*x1), xmask)
    tmp75 = tl.load(in_ptr4 + (48 + x0 + 64*x1), xmask)
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
    tmp27 = -tmp26
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tmp28 / tmp34
    tmp36 = tmp29 / tmp34
    tmp37 = triton_helpers.maximum(tmp35, tmp36)
    tmp38 = tmp31 / tmp34
    tmp39 = triton_helpers.maximum(tmp37, tmp38)
    tmp40 = tmp33 / tmp34
    tmp41 = triton_helpers.maximum(tmp39, tmp40)
    tmp42 = 1.0
    tmp43 = tmp42 - tmp41
    tmp44 = tmp43 * tmp43
    tmp45 = tmp27 * tmp44
    tmp47 = tl_math.exp(tmp46)
    tmp49 = tl_math.exp(tmp48)
    tmp50 = tmp47 + tmp49
    tmp52 = tl_math.exp(tmp51)
    tmp53 = tmp50 + tmp52
    tmp55 = tl_math.exp(tmp54)
    tmp56 = tmp53 + tmp55
    tmp57 = tl_math.log(tmp56)
    tmp58 = tmp46 - tmp57
    tmp59 = tmp58 * tmp13
    tmp60 = tmp48 - tmp57
    tmp61 = tmp60 * tmp16
    tmp62 = tmp59 + tmp61
    tmp63 = tmp51 - tmp57
    tmp64 = tmp63 * tmp20
    tmp65 = tmp62 + tmp64
    tmp66 = tmp54 - tmp57
    tmp67 = tmp66 * tmp24
    tmp68 = tmp65 + tmp67
    tmp69 = -tmp68
    tmp72 = tmp70 + tmp71
    tmp74 = tmp72 + tmp73
    tmp76 = tmp74 + tmp75
    tmp77 = tmp70 / tmp76
    tmp78 = tmp71 / tmp76
    tmp79 = triton_helpers.maximum(tmp77, tmp78)
    tmp80 = tmp73 / tmp76
    tmp81 = triton_helpers.maximum(tmp79, tmp80)
    tmp82 = tmp75 / tmp76
    tmp83 = triton_helpers.maximum(tmp81, tmp82)
    tmp84 = tmp42 - tmp83
    tmp85 = tmp84 * tmp84
    tmp86 = tmp69 * tmp85
    tmp87 = tmp45 + tmp86
    tl.store(in_out_ptr0 + (x2), tmp87, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eu/ceu6ibf6jul6jlhcz75dfdd3b5vkbtboetigwqookzn3q2l3bdr5.py
# Topologically Sorted Source Nodes: [gt, type_1], Original ATen: [aten.gt, aten._to_copy]
# Source node to ATen node mapping:
#   gt => gt
#   type_1 => convert_element_type
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt, torch.float32), kwargs = {})
triton_poi_fused__to_copy_gt_2 = async_compile.triton('triton_poi_fused__to_copy_gt_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_gt_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_gt_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


cpp_fused_eq_sum_3 = async_compile.cpp_pybinding(['const float*', 'float*', 'bool*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    {
        {
            float tmp_acc0 = 0;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
    {
        {
            {
                auto tmp0 = out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 == tmp1;
                out_ptr1[static_cast<int64_t>(0L)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [softmax, loss_stage1], Original ATen: [aten._softmax, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__softmax_0.run(arg1_1, buf0, buf2, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [softmax_1, loss_stage2], Original ATen: [aten._softmax, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__softmax_0.run(arg2_1, buf1, buf4, 256, grid=grid(256), stream=stream0)
        del arg2_1
        buf3 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf6 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [softmax, max_1, softmax_1, max_2, loss_stage1, sub, exponential_term_stage1, loss_stage1_1, loss_stage2, sub_1, exponential_term_stage2, loss_stage2_1, loss], Original ATen: [aten._softmax, aten.max, aten._log_softmax, aten.mul, aten.sum, aten.neg, aten.rsub, aten.pow, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__softmax_add_max_mul_neg_pow_rsub_sum_1.run(buf6, buf2, arg0_1, buf0, buf4, buf1, 64, grid=grid(64), stream=stream0)
        del buf0
        del buf1
        del buf2
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [gt, type_1], Original ATen: [aten.gt, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_gt_2.run(arg0_1, buf7, 256, grid=grid(256), stream=stream0)
        del arg0_1
    buf8 = empty_strided_cpu((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
    buf8.copy_(buf7, False)
    del buf7
    buf9 = empty_strided_cpu((), (), torch.float32)
    buf10 = empty_strided_cpu((), (), torch.bool)
    cpp_fused_eq_sum_3(buf8, buf9, buf10)
    return (buf6, buf9, buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
