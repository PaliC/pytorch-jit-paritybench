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


cpp_fused_add_affine_grid_generator_copy_mul_repeat_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(3L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(3L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp8 = in_ptr0[static_cast<int64_t>(x0)];
                            auto tmp15 = in_ptr1[static_cast<int64_t>(x0)];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int32_t>(tmp0);
                            auto tmp2 = static_cast<int32_t>(1);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp4 = x2;
                            auto tmp5 = c10::convert<int32_t>(tmp4);
                            auto tmp6 = static_cast<int32_t>(2);
                            auto tmp7 = tmp5 == tmp6;
                            auto tmp9 = static_cast<float>(-5.25);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = static_cast<float>(2.75);
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            auto tmp13 = static_cast<int32_t>(0);
                            auto tmp14 = tmp2 == tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp9);
                            auto tmp17 = decltype(tmp16)(tmp16 + tmp11);
                            auto tmp18 = static_cast<int64_t>(0);
                            auto tmp19 = c10::convert<int64_t>(tmp4);
                            auto tmp20 = tmp18 == tmp19;
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = static_cast<float>(0.0);
                            auto tmp23 = tmp20 ? tmp21 : tmp22;
                            auto tmp24 = tmp7 ? tmp17 : tmp23;
                            auto tmp25 = static_cast<int64_t>(1);
                            auto tmp26 = tmp25 == tmp19;
                            auto tmp27 = tmp26 ? tmp21 : tmp22;
                            auto tmp28 = tmp14 ? tmp24 : tmp27;
                            auto tmp29 = tmp7 ? tmp12 : tmp28;
                            auto tmp30 = tmp1 == tmp13;
                            auto tmp31 = c10::convert<int64_t>(tmp0);
                            auto tmp32 = tmp31 == tmp19;
                            auto tmp33 = tmp32 ? tmp21 : tmp22;
                            auto tmp34 = tmp30 ? tmp24 : tmp33;
                            auto tmp35 = tmp3 ? tmp29 : tmp34;
                            out_ptr0[static_cast<int64_t>(x2 + 3L*x1 + 9L*x0)] = tmp35;
                        }
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(16L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(2L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp46 = out_ptr0[static_cast<int64_t>(3L*x2 + 9L*x0)];
                            auto tmp88 = out_ptr0[static_cast<int64_t>(1L + 3L*x2 + 9L*x0)];
                            auto tmp132 = out_ptr0[static_cast<int64_t>(2L + 3L*x2 + 9L*x0)];
                            auto tmp0 = static_cast<int64_t>(0);
                            auto tmp1 = static_cast<int64_t>(1);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = (static_cast<int64_t>(x1) % static_cast<int64_t>(4L));
                                auto tmp5 = c10::convert<float>(tmp4);
                                auto tmp6 = static_cast<float>(2.0);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = static_cast<float>(0.5);
                                auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                                auto tmp10 = static_cast<float>(-0.75);
                                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                                auto tmp12 = 3L + ((-1L)*((static_cast<int64_t>(x1) % static_cast<int64_t>(4L))));
                                auto tmp13 = c10::convert<float>(tmp12);
                                auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                                auto tmp15 = static_cast<float>(0.75);
                                auto tmp16 = decltype(tmp15)(tmp15 - tmp14);
                                auto tmp17 = tmp7 ? tmp11 : tmp16;
                                return tmp17;
                            }
                            ;
                            auto tmp18 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            auto tmp19 = static_cast<int64_t>(-1);
                            auto tmp20 = tmp19 >= tmp0;
                            auto tmp21 = tmp19 < tmp1;
                            auto tmp22 = tmp20 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(4L));
                                auto tmp25 = c10::convert<float>(tmp24);
                                auto tmp26 = static_cast<float>(2.0);
                                auto tmp27 = tmp25 < tmp26;
                                auto tmp28 = static_cast<float>(0.5);
                                auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                                auto tmp30 = static_cast<float>(-0.75);
                                auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                                auto tmp32 = 3L + ((-1L)*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(4L))));
                                auto tmp33 = c10::convert<float>(tmp32);
                                auto tmp34 = decltype(tmp33)(tmp33 * tmp28);
                                auto tmp35 = static_cast<float>(0.75);
                                auto tmp36 = decltype(tmp35)(tmp35 - tmp34);
                                auto tmp37 = tmp27 ? tmp31 : tmp36;
                                return tmp37;
                            }
                            ;
                            auto tmp38 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp39 = decltype(tmp18)(tmp18 + tmp38);
                            auto tmp40 = static_cast<int64_t>(-2);
                            auto tmp41 = tmp40 >= tmp0;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = static_cast<float>(1.0);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp45 = decltype(tmp39)(tmp39 + tmp44);
                            auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                            auto tmp48 = tmp1 < tmp1;
                            auto tmp49 = [&]
                            {
                                auto tmp50 = (static_cast<int64_t>(x1) % static_cast<int64_t>(4L));
                                auto tmp51 = c10::convert<float>(tmp50);
                                auto tmp52 = static_cast<float>(2.0);
                                auto tmp53 = tmp51 < tmp52;
                                auto tmp54 = static_cast<float>(0.5);
                                auto tmp55 = decltype(tmp51)(tmp51 * tmp54);
                                auto tmp56 = static_cast<float>(-0.75);
                                auto tmp57 = decltype(tmp55)(tmp55 + tmp56);
                                auto tmp58 = 3L + ((-1L)*((static_cast<int64_t>(x1) % static_cast<int64_t>(4L))));
                                auto tmp59 = c10::convert<float>(tmp58);
                                auto tmp60 = decltype(tmp59)(tmp59 * tmp54);
                                auto tmp61 = static_cast<float>(0.75);
                                auto tmp62 = decltype(tmp61)(tmp61 - tmp60);
                                auto tmp63 = tmp53 ? tmp57 : tmp62;
                                return tmp63;
                            }
                            ;
                            auto tmp64 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                            auto tmp65 = tmp0 >= tmp0;
                            auto tmp66 = tmp65 & tmp2;
                            auto tmp67 = [&]
                            {
                                auto tmp68 = c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(4L));
                                auto tmp69 = c10::convert<float>(tmp68);
                                auto tmp70 = static_cast<float>(2.0);
                                auto tmp71 = tmp69 < tmp70;
                                auto tmp72 = static_cast<float>(0.5);
                                auto tmp73 = decltype(tmp69)(tmp69 * tmp72);
                                auto tmp74 = static_cast<float>(-0.75);
                                auto tmp75 = decltype(tmp73)(tmp73 + tmp74);
                                auto tmp76 = 3L + ((-1L)*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(4L))));
                                auto tmp77 = c10::convert<float>(tmp76);
                                auto tmp78 = decltype(tmp77)(tmp77 * tmp72);
                                auto tmp79 = static_cast<float>(0.75);
                                auto tmp80 = decltype(tmp79)(tmp79 - tmp78);
                                auto tmp81 = tmp71 ? tmp75 : tmp80;
                                return tmp81;
                            }
                            ;
                            auto tmp82 = tmp66 ? tmp67() : static_cast<decltype(tmp67())>(0.0);
                            auto tmp83 = decltype(tmp64)(tmp64 + tmp82);
                            auto tmp84 = [&]
                            {
                                auto tmp85 = static_cast<float>(1.0);
                                return tmp85;
                            }
                            ;
                            auto tmp86 = tmp20 ? tmp84() : static_cast<decltype(tmp84())>(0.0);
                            auto tmp87 = decltype(tmp83)(tmp83 + tmp86);
                            auto tmp89 = decltype(tmp87)(tmp87 * tmp88);
                            auto tmp90 = decltype(tmp47)(tmp47 + tmp89);
                            auto tmp91 = static_cast<int64_t>(2);
                            auto tmp92 = tmp91 < tmp1;
                            auto tmp93 = [&]
                            {
                                auto tmp94 = (static_cast<int64_t>(x1) % static_cast<int64_t>(4L));
                                auto tmp95 = c10::convert<float>(tmp94);
                                auto tmp96 = static_cast<float>(2.0);
                                auto tmp97 = tmp95 < tmp96;
                                auto tmp98 = static_cast<float>(0.5);
                                auto tmp99 = decltype(tmp95)(tmp95 * tmp98);
                                auto tmp100 = static_cast<float>(-0.75);
                                auto tmp101 = decltype(tmp99)(tmp99 + tmp100);
                                auto tmp102 = 3L + ((-1L)*((static_cast<int64_t>(x1) % static_cast<int64_t>(4L))));
                                auto tmp103 = c10::convert<float>(tmp102);
                                auto tmp104 = decltype(tmp103)(tmp103 * tmp98);
                                auto tmp105 = static_cast<float>(0.75);
                                auto tmp106 = decltype(tmp105)(tmp105 - tmp104);
                                auto tmp107 = tmp97 ? tmp101 : tmp106;
                                return tmp107;
                            }
                            ;
                            auto tmp108 = tmp92 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                            auto tmp109 = tmp1 >= tmp0;
                            auto tmp110 = tmp109 & tmp48;
                            auto tmp111 = [&]
                            {
                                auto tmp112 = c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(4L));
                                auto tmp113 = c10::convert<float>(tmp112);
                                auto tmp114 = static_cast<float>(2.0);
                                auto tmp115 = tmp113 < tmp114;
                                auto tmp116 = static_cast<float>(0.5);
                                auto tmp117 = decltype(tmp113)(tmp113 * tmp116);
                                auto tmp118 = static_cast<float>(-0.75);
                                auto tmp119 = decltype(tmp117)(tmp117 + tmp118);
                                auto tmp120 = 3L + ((-1L)*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(4L))));
                                auto tmp121 = c10::convert<float>(tmp120);
                                auto tmp122 = decltype(tmp121)(tmp121 * tmp116);
                                auto tmp123 = static_cast<float>(0.75);
                                auto tmp124 = decltype(tmp123)(tmp123 - tmp122);
                                auto tmp125 = tmp115 ? tmp119 : tmp124;
                                return tmp125;
                            }
                            ;
                            auto tmp126 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(0.0);
                            auto tmp127 = decltype(tmp108)(tmp108 + tmp126);
                            auto tmp128 = [&]
                            {
                                auto tmp129 = static_cast<float>(1.0);
                                return tmp129;
                            }
                            ;
                            auto tmp130 = tmp65 ? tmp128() : static_cast<decltype(tmp128())>(0.0);
                            auto tmp131 = decltype(tmp127)(tmp127 + tmp130);
                            auto tmp133 = decltype(tmp131)(tmp131 * tmp132);
                            auto tmp134 = decltype(tmp90)(tmp90 + tmp133);
                            out_ptr1[static_cast<int64_t>(x2 + 2L*x1 + 32L*x0)] = tmp134;
                        }
                    }
                }
            }
        }
    }
}
''')


# kernel path: inductor_cache/nj/cnjv33q7dutsxaumr2kvxy7avewfv7ndudg6ju2l5ezbz2osljzq.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   x => add_10, add_11, add_12, add_6, add_7, add_8, add_9, floor, floor_1, full_default_11, full_default_14, full_default_5, full_default_8, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index, index_1, index_2, index_3, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_7, mul_8, mul_9, sub_10, sub_11, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9, where_11, where_14, where_5, where_8
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, 2.0), kwargs = {})
#   %add_6 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, 1.5), kwargs = {})
#   %floor : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_6,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, 2.0), kwargs = {})
#   %add_7 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, 1.5), kwargs = {})
#   %floor_1 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_7,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %lt_3), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_2, %logical_and), kwargs = {})
#   %logical_and_2 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %logical_and_1), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_4, %where_3]), kwargs = {})
#   %add_8 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor, 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %add_6), kwargs = {})
#   %add_9 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_1, 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %add_7), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %sub_5), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %mul_9, %full_default_5), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %where_5), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_8, 0), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_8, 4), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_3, %lt_5), kwargs = {})
#   %logical_and_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_4, %logical_and_3), kwargs = {})
#   %logical_and_5 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_2, %logical_and_4), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_7, %where_6]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %floor), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %add_7), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %sub_7), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_5, %mul_10, %full_default_8), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_1, %where_8), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_9, 0), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_9, 4), kwargs = {})
#   %logical_and_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_5, %lt_7), kwargs = {})
#   %logical_and_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_6, %logical_and_6), kwargs = {})
#   %logical_and_8 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_4, %logical_and_7), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_10, %where_9]), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %add_6), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %floor_1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %sub_9), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_8, %mul_11, %full_default_11), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %where_11), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_15), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_8, 0), kwargs = {})
#   %lt_8 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_8, 4), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_9, 0), kwargs = {})
#   %lt_9 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_9, 4), kwargs = {})
#   %logical_and_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_7, %lt_9), kwargs = {})
#   %logical_and_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_8, %logical_and_9), kwargs = {})
#   %logical_and_11 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_6, %logical_and_10), kwargs = {})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view_5, %view_6, %where_13, %where_12]), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %floor), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %floor_1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %sub_11), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_14 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %mul_12, %full_default_14), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_3, %where_14), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_16), kwargs = {})
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_poi_fused_grid_sampler_2d_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex // 64
    x4 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 4.0
    tmp9 = tmp5 < tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp3
    tmp13 = libdevice.floor(tmp12)
    tmp14 = tmp13 >= tmp6
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp9 & tmp16
    tmp18 = tmp7 & tmp17
    tmp19 = tmp13.to(tl.int64)
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.full([XBLOCK], 4, tl.int32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp21 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp21)
    tl.device_assert(((0 <= tmp25) & (tmp25 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp25 < 4")
    tmp27 = tmp5.to(tl.int64)
    tmp28 = tl.where(tmp18, tmp27, tmp20)
    tmp29 = tmp28 + tmp22
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tl.device_assert(((0 <= tmp31) & (tmp31 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp31 < 4")
    tmp33 = tl.load(in_ptr1 + (tmp31 + 4*tmp25 + 16*x4), xmask, eviction_policy='evict_last')
    tmp34 = 1.0
    tmp35 = tmp5 + tmp34
    tmp36 = tmp35 - tmp4
    tmp37 = tmp13 + tmp34
    tmp38 = tmp37 - tmp12
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp18, tmp39, tmp6)
    tmp41 = tmp33 * tmp40
    tmp42 = tmp35 >= tmp6
    tmp43 = tmp35 < tmp8
    tmp44 = tmp43 & tmp16
    tmp45 = tmp42 & tmp44
    tmp46 = tl.where(tmp45, tmp19, tmp20)
    tmp47 = tmp46 + tmp22
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp49 < 4")
    tmp51 = tmp35.to(tl.int64)
    tmp52 = tl.where(tmp45, tmp51, tmp20)
    tmp53 = tmp52 + tmp22
    tmp54 = tmp52 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp52)
    tl.device_assert(((0 <= tmp55) & (tmp55 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tl.load(in_ptr1 + (tmp55 + 4*tmp49 + 16*x4), xmask, eviction_policy='evict_last')
    tmp58 = tmp4 - tmp5
    tmp59 = tmp58 * tmp38
    tmp60 = tl.where(tmp45, tmp59, tmp6)
    tmp61 = tmp57 * tmp60
    tmp62 = tmp37 >= tmp6
    tmp63 = tmp37 < tmp8
    tmp64 = tmp62 & tmp63
    tmp65 = tmp9 & tmp64
    tmp66 = tmp7 & tmp65
    tmp67 = tmp37.to(tl.int64)
    tmp68 = tl.where(tmp66, tmp67, tmp20)
    tmp69 = tmp68 + tmp22
    tmp70 = tmp68 < 0
    tmp71 = tl.where(tmp70, tmp69, tmp68)
    tl.device_assert(((0 <= tmp71) & (tmp71 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp71 < 4")
    tmp73 = tl.where(tmp66, tmp27, tmp20)
    tmp74 = tmp73 + tmp22
    tmp75 = tmp73 < 0
    tmp76 = tl.where(tmp75, tmp74, tmp73)
    tl.device_assert(((0 <= tmp76) & (tmp76 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp76 < 4")
    tmp78 = tl.load(in_ptr1 + (tmp76 + 4*tmp71 + 16*x4), xmask, eviction_policy='evict_last')
    tmp79 = tmp12 - tmp13
    tmp80 = tmp36 * tmp79
    tmp81 = tl.where(tmp66, tmp80, tmp6)
    tmp82 = tmp78 * tmp81
    tmp83 = tmp43 & tmp64
    tmp84 = tmp42 & tmp83
    tmp85 = tl.where(tmp84, tmp67, tmp20)
    tmp86 = tmp85 + tmp22
    tmp87 = tmp85 < 0
    tmp88 = tl.where(tmp87, tmp86, tmp85)
    tl.device_assert(((0 <= tmp88) & (tmp88 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp88 < 4")
    tmp90 = tl.where(tmp84, tmp51, tmp20)
    tmp91 = tmp90 + tmp22
    tmp92 = tmp90 < 0
    tmp93 = tl.where(tmp92, tmp91, tmp90)
    tl.device_assert(((0 <= tmp93) & (tmp93 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp93 < 4")
    tmp95 = tl.load(in_ptr1 + (tmp93 + 4*tmp88 + 16*x4), xmask, eviction_policy='evict_last')
    tmp96 = tmp58 * tmp79
    tmp97 = tl.where(tmp84, tmp96, tmp6)
    tmp98 = tmp95 * tmp97
    tmp99 = tmp41 + tmp61
    tmp100 = tmp99 + tmp82
    tmp101 = tmp100 + tmp98
    tl.store(in_out_ptr0 + (x3), tmp101, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    # Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
    buf0 = torch.ops.aten.rand.default([4], device=device(type='cpu'), pin_memory=False)
    buf1 = buf0
    del buf0
    # Topologically Sorted Source Nodes: [rand_1], Original ATen: [aten.rand]
    buf2 = torch.ops.aten.rand.default([4], device=device(type='cpu'), pin_memory=False)
    buf3 = buf2
    del buf2
    buf4 = empty_strided_cpu((4, 3, 3), (9, 3, 1), torch.float32)
    buf6 = empty_strided_cpu((4, 16, 2), (32, 2, 1), torch.float32)
    cpp_fused_add_affine_grid_generator_copy_mul_repeat_0(buf3, buf1, buf4, buf6)
    del buf1
    del buf3
    del buf4
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf7 = empty_strided_cuda((4, 4, 4, 2), (32, 8, 2, 1), torch.float32)
        buf7.copy_(reinterpret_tensor(buf6, (4, 4, 4, 2), (32, 8, 2, 1), 0), False)
        del buf6
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_1.run(buf12, buf7, arg0_1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del buf7
    return (buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
