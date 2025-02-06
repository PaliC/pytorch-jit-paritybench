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


cpp_fused__to_copy_stack_0 = async_compile.cpp_pybinding(['const float*', 'const int64_t*', 'int64_t*', 'int64_t*', 'int64_t*', 'int64_t*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const int64_t* in_ptr1,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       int64_t* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4096L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = 16L*((static_cast<int64_t>(x0) % static_cast<int64_t>(64L)));
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    out_ptr0[static_cast<int64_t>(4L*x0)] = tmp1;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4096L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = 16L*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(64L)));
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    out_ptr1[static_cast<int64_t>(4L*x0)] = tmp1;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4096L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = 16L*((static_cast<int64_t>(x0) % static_cast<int64_t>(64L)));
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    out_ptr2[static_cast<int64_t>(4L*x0)] = tmp1;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4096L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = 16L*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(64L)));
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    out_ptr3[static_cast<int64_t>(4L*x0)] = tmp1;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(36864L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 4L*((static_cast<int64_t>(x0) % static_cast<int64_t>(9L))))];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 4L*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(9L))))];
                        auto tmp1 = c10::convert<double>(tmp0);
                        auto tmp3 = c10::convert<double>(tmp2);
                        auto tmp4 = decltype(tmp1)(tmp1 + tmp3);
                        auto tmp5 = c10::convert<float>(tmp4);
                        out_ptr4[static_cast<int64_t>(x1 + 4L*x0)] = tmp5;
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
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (9, 4), (4, 1))
    buf4 = empty_strided_cpu((4096, 4), (4, 1), torch.int64)
    buf0 = reinterpret_tensor(buf4, (4096, 1), (4, 1), 0)  # alias
    buf1 = reinterpret_tensor(buf4, (4096, 1), (4, 1), 1)  # alias
    buf2 = reinterpret_tensor(buf4, (4096, 1), (4, 1), 2)  # alias
    buf3 = reinterpret_tensor(buf4, (4096, 1), (4, 1), 3)  # alias
    buf5 = empty_strided_cpu((36864, 4), (4, 1), torch.float32)
    cpp_fused__to_copy_stack_0(arg0_1, buf4, buf0, buf1, buf2, buf3, buf5)
    del arg0_1
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((9, 4), (4, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
