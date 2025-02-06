# AOT ID: ['61_forward']
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
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 2), (8, 2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_ft], Original ATen: [aten._fft_r2c]
        buf0 = torch.ops.aten._fft_r2c.default(primals_1, [3], 0, True)
        del primals_1
        buf1 = buf0
        del buf0
        # Topologically Sorted Source Nodes: [out_ft], Original ATen: [aten.zeros]
        buf2 = torch.ops.aten.full.default([4, 4, 4, 3], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf3 = buf2
        del buf2
        # Topologically Sorted Source Nodes: [a], Original ATen: [aten.slice]
        buf4 = torch.ops.aten.slice.Tensor(buf1, 3, 0, 2)
        buf5 = buf4
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.unsqueeze]
        buf6 = torch.ops.aten.unsqueeze.default(buf5, 4)
        buf7 = buf6
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.permute]
        buf8 = torch.ops.aten.permute.default(buf7, [0, 1, 4, 3, 2])
        buf9 = buf8
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.unsqueeze]
        buf10 = torch.ops.aten.unsqueeze.default(primals_2, 3)
        buf11 = buf10
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.unsqueeze]
        buf12 = torch.ops.aten.unsqueeze.default(buf11, 4)
        buf13 = buf12
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.permute]
        buf14 = torch.ops.aten.permute.default(buf13, [3, 4, 1, 2, 0])
        buf15 = buf14
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.permute]
        buf16 = torch.ops.aten.permute.default(buf9, [3, 0, 1, 4, 2])
        buf17 = buf16
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.view]
        buf18 = torch.ops.aten.reshape.default(buf17, [2, 16, 4])
        buf19 = buf18
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.permute]
        buf20 = torch.ops.aten.permute.default(buf15, [3, 4, 2, 0, 1])
        buf21 = buf20
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.view]
        buf22 = torch.ops.aten.reshape.default(buf21, [2, 4, 4])
        buf23 = buf22
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        buf24 = torch.ops.aten.bmm.default(buf19, buf23)
        del buf10
        del buf11
        del buf12
        del buf13
        del buf14
        del buf15
        del buf20
        del buf21
        del buf22
        del buf23
        del primals_2
        buf25 = buf24
        del buf24
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.view]
        buf26 = torch.ops.aten.reshape.default(buf25, [2, 4, 4, 1, 4])
        buf27 = buf26
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.permute]
        buf28 = torch.ops.aten.permute.default(buf27, [1, 2, 4, 0, 3])
        buf29 = buf28
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.view]
        buf30 = torch.ops.aten.reshape.default(buf29, [4, 4, 4, 2])
        buf31 = buf30
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.slice]
        buf32 = torch.ops.aten.slice.Tensor(buf3, 3, 0, 2)
        buf33 = buf32
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.copy]
        buf34 = torch.ops.aten.copy.default(buf33, buf31)
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        del buf30
        del buf31
        del buf32
        del buf33
        buf35 = buf34
        del buf34
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf36 = torch.ops.aten.slice_scatter.default(buf3, buf35, 3, 0, 2)
        del buf3
        del buf35
        buf37 = buf36
        del buf36
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._fft_c2r]
        buf38 = torch.ops.aten._fft_c2r.default(buf37, [3], 2, 4)
        del buf37
        buf39 = buf38
        del buf38
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose]
        buf40 = torch.ops.aten.permute.default(buf19, [0, 2, 1])
        buf41 = buf40
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf42 = torch.ops.aten._conj.default(buf41)
        buf43 = buf42
    return (buf39, buf43, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 2), (8, 2, 1), device='cuda:0', dtype=torch.complex64)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
