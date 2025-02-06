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


cpp_fused_lift_fresh_prod_0 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(4);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp8;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    buf0 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_0(buf0)
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    fn = lambda: call([])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
