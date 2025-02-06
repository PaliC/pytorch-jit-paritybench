# AOT ID: ['1_inference']
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


async_compile.wait(globals())
del async_compile

def call(args):
    # Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
    buf0 = torch.ops.aten.rand.default([7, 4], device=device(type='cpu'), pin_memory=False)
    buf1 = buf0
    del buf0
    # Topologically Sorted Source Nodes: [randint], Original ATen: [aten.randint]
    buf2 = torch.ops.aten.randint.low(0, 4, [7], device=device(type='cpu'), pin_memory=False)
    buf3 = buf2
    del buf2
    # Topologically Sorted Source Nodes: [rand_1], Original ATen: [aten.rand]
    buf4 = torch.ops.aten.rand.default([7], device=device(type='cpu'), pin_memory=False)
    buf5 = buf4
    del buf4
    # Topologically Sorted Source Nodes: [rand_2], Original ATen: [aten.rand]
    buf6 = torch.ops.aten.rand.default([7, 4], device=device(type='cpu'), pin_memory=False)
    buf7 = buf6
    del buf6
    # Topologically Sorted Source Nodes: [randint_1], Original ATen: [aten.randint]
    buf8 = torch.ops.aten.randint.low(0, 4, [7], device=device(type='cpu'), pin_memory=False)
    buf9 = buf8
    del buf8
    # Topologically Sorted Source Nodes: [rand_3], Original ATen: [aten.rand]
    buf10 = torch.ops.aten.rand.default([7], device=device(type='cpu'), pin_memory=False)
    buf11 = buf10
    del buf10
    # Topologically Sorted Source Nodes: [rand_4], Original ATen: [aten.rand]
    buf12 = torch.ops.aten.rand.default([7, 4], device=device(type='cpu'), pin_memory=False)
    buf13 = buf12
    del buf12
    # Topologically Sorted Source Nodes: [randint_2], Original ATen: [aten.randint]
    buf14 = torch.ops.aten.randint.low(0, 4, [7], device=device(type='cpu'), pin_memory=False)
    buf15 = buf14
    del buf14
    # Topologically Sorted Source Nodes: [rand_5], Original ATen: [aten.rand]
    buf16 = torch.ops.aten.rand.default([7], device=device(type='cpu'), pin_memory=False)
    buf17 = buf16
    del buf16
    # Topologically Sorted Source Nodes: [rand_6], Original ATen: [aten.rand]
    buf18 = torch.ops.aten.rand.default([7, 4], device=device(type='cpu'), pin_memory=False)
    buf19 = buf18
    del buf18
    # Topologically Sorted Source Nodes: [randint_3], Original ATen: [aten.randint]
    buf20 = torch.ops.aten.randint.low(0, 4, [7], device=device(type='cpu'), pin_memory=False)
    buf21 = buf20
    del buf20
    # Topologically Sorted Source Nodes: [rand_7], Original ATen: [aten.rand]
    buf22 = torch.ops.aten.rand.default([7], device=device(type='cpu'), pin_memory=False)
    buf23 = buf22
    return (buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    fn = lambda: call([])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
