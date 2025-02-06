# AOT ID: ['2_inference']
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


cpp_fused_max_pool2d_with_indices_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(16L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(4L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = (-1L) + x1;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = static_cast<int64_t>(0);
                            auto tmp3 = tmp1 >= tmp2;
                            auto tmp4 = static_cast<int64_t>(4);
                            auto tmp5 = tmp1 < tmp4;
                            auto tmp6 = tmp3 & tmp5;
                            auto tmp7 = (-1L) + x2;
                            auto tmp8 = c10::convert<int64_t>(tmp7);
                            auto tmp9 = tmp8 >= tmp2;
                            auto tmp10 = tmp8 < tmp4;
                            auto tmp11 = tmp9 & tmp10;
                            auto tmp12 = tmp6 & tmp11;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = in_ptr0[static_cast<int64_t>((-5L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp12 ? tmp13() : -std::numeric_limits<decltype(tmp13())>::infinity();
                            auto tmp16 = x2;
                            auto tmp17 = c10::convert<int64_t>(tmp16);
                            auto tmp18 = tmp17 >= tmp2;
                            auto tmp19 = tmp17 < tmp4;
                            auto tmp20 = tmp18 & tmp19;
                            auto tmp21 = tmp6 & tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr0[static_cast<int64_t>((-4L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp21 ? tmp22() : -std::numeric_limits<decltype(tmp22())>::infinity();
                            auto tmp25 = max_propagate_nan(tmp24, tmp15);
                            auto tmp26 = 1L + x2;
                            auto tmp27 = c10::convert<int64_t>(tmp26);
                            auto tmp28 = tmp27 >= tmp2;
                            auto tmp29 = tmp27 < tmp4;
                            auto tmp30 = tmp28 & tmp29;
                            auto tmp31 = tmp6 & tmp30;
                            auto tmp32 = [&]
                            {
                                auto tmp33 = in_ptr0[static_cast<int64_t>((-3L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp31 ? tmp32() : -std::numeric_limits<decltype(tmp32())>::infinity();
                            auto tmp35 = max_propagate_nan(tmp34, tmp25);
                            auto tmp36 = x1;
                            auto tmp37 = c10::convert<int64_t>(tmp36);
                            auto tmp38 = tmp37 >= tmp2;
                            auto tmp39 = tmp37 < tmp4;
                            auto tmp40 = tmp38 & tmp39;
                            auto tmp41 = tmp40 & tmp11;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr0[static_cast<int64_t>((-1L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp35);
                            auto tmp46 = tmp40 & tmp20;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr0[static_cast<int64_t>(x2 + 4L*x1 + 16L*x0)];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = tmp40 & tmp30;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr0[static_cast<int64_t>(1L + x2 + 4L*x1 + 16L*x0)];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp51 ? tmp52() : -std::numeric_limits<decltype(tmp52())>::infinity();
                            auto tmp55 = max_propagate_nan(tmp54, tmp50);
                            auto tmp56 = 1L + x1;
                            auto tmp57 = c10::convert<int64_t>(tmp56);
                            auto tmp58 = tmp57 >= tmp2;
                            auto tmp59 = tmp57 < tmp4;
                            auto tmp60 = tmp58 & tmp59;
                            auto tmp61 = tmp60 & tmp11;
                            auto tmp62 = [&]
                            {
                                auto tmp63 = in_ptr0[static_cast<int64_t>(3L + x2 + 4L*x1 + 16L*x0)];
                                return tmp63;
                            }
                            ;
                            auto tmp64 = tmp61 ? tmp62() : -std::numeric_limits<decltype(tmp62())>::infinity();
                            auto tmp65 = max_propagate_nan(tmp64, tmp55);
                            auto tmp66 = tmp60 & tmp20;
                            auto tmp67 = [&]
                            {
                                auto tmp68 = in_ptr0[static_cast<int64_t>(4L + x2 + 4L*x1 + 16L*x0)];
                                return tmp68;
                            }
                            ;
                            auto tmp69 = tmp66 ? tmp67() : -std::numeric_limits<decltype(tmp67())>::infinity();
                            auto tmp70 = max_propagate_nan(tmp69, tmp65);
                            auto tmp71 = tmp60 & tmp30;
                            auto tmp72 = [&]
                            {
                                auto tmp73 = in_ptr0[static_cast<int64_t>(5L + x2 + 4L*x1 + 16L*x0)];
                                return tmp73;
                            }
                            ;
                            auto tmp74 = tmp71 ? tmp72() : -std::numeric_limits<decltype(tmp72())>::infinity();
                            auto tmp75 = max_propagate_nan(tmp74, tmp70);
                            out_ptr0[static_cast<int64_t>(x2 + 4L*x1 + 16L*x0)] = tmp75;
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    buf0 = empty_strided_cpu((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
    cpp_fused_max_pool2d_with_indices_0(arg0_1, buf0)
    return (reinterpret_tensor(arg1_1, (4, 4, 4, 4), (64, 4, 1, 16), 0), reinterpret_tensor(arg0_1, (4, 4, 4, 4), (64, 4, 1, 16), 0), reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 4, 1, 16), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
