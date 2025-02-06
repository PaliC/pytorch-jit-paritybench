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


cpp_fused_index_stack_0 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                        auto tmp1 = 4L;
                        auto tmp2 = c10::convert<int64_t>(tmp1);
                        auto tmp3 = decltype(tmp0)(tmp0 + tmp2);
                        auto tmp4 = tmp0 < 0;
                        auto tmp5 = tmp4 ? tmp3 : tmp0;
                        auto tmp6 = tmp5;
                        auto tmp7 = c10::convert<int64_t>(tmp6);
                        AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 4L), "index out of bounds: 0 <= tmp7 < 4L");
                        auto tmp9 = in_ptr1[static_cast<int64_t>(x1 + 64L*tmp5)];
                        out_ptr0[static_cast<int64_t>(x1 + 64L*x0)] = tmp9;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr2[static_cast<int64_t>(x0)];
                    out_ptr1[static_cast<int64_t>(x0)] = tmp0;
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
    # Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
    buf0 = torch.ops.aten.rand.default([4], device=device(type='cpu'), pin_memory=False)
    buf1 = buf0
    del buf0
    # Topologically Sorted Source Nodes: [perm], Original ATen: [aten.sort]
    buf2 = torch.ops.aten.sort.stable(buf1, stable=False, dim=0, descending=False)
    del buf1
    buf4 = buf2[1]
    del buf2
    buf7 = empty_strided_cpu((8, 4, 4, 4), (64, 16, 4, 1), torch.float32)
    buf5 = reinterpret_tensor(buf7, (4, 4, 4, 4), (64, 16, 4, 1), 0)  # alias
    buf6 = reinterpret_tensor(buf7, (4, 4, 4, 4), (64, 16, 4, 1), 256)  # alias
    cpp_fused_index_stack_0(buf4, arg0_1, arg1_1, buf5, buf6)
    del arg0_1
    del arg1_1
    return (reinterpret_tensor(buf7, (2, 4, 4, 4, 4), (256, 64, 16, 4, 1), 0), )


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
