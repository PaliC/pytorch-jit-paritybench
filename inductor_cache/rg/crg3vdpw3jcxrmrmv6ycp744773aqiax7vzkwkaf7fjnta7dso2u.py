# AOT ID: ['7_inference']
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [out_ft], Original ATen: [aten.zeros]
        buf0 = torch.ops.aten.full.default([4, 4, 4, 3], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.select]
        buf2 = torch.ops.aten.select.int(buf1, 3, 0)
        buf3 = buf2
        # Topologically Sorted Source Nodes: [xq_ft_], Original ATen: [aten.zeros]
        buf4 = torch.ops.aten.full.default([4, 4, 4, 2], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf5 = buf4
        del buf4
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.select]
        buf6 = torch.ops.aten.select.int(buf5, 3, 0)
        buf7 = buf6
        # Topologically Sorted Source Nodes: [xq_ft], Original ATen: [aten._fft_r2c]
        buf8 = torch.ops.aten._fft_r2c.default(reinterpret_tensor(arg0_1, (4, 4, 4, 4), (64, 1, 4, 16), 0), [3], 0, True)
        del arg0_1
        buf9 = buf8
        del buf8
        # Topologically Sorted Source Nodes: [getitem], Original ATen: [aten.select]
        buf10 = torch.ops.aten.select.int(buf9, 3, 0)
        buf11 = buf10
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.copy]
        buf12 = torch.ops.aten.copy.default(buf7, buf11)
        del buf10
        del buf11
        del buf6
        del buf7
        buf13 = buf12
        del buf12
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf14 = torch.ops.aten.select_scatter.default(buf5, buf13, 3, 0)
        del buf13
        del buf5
        buf15 = buf14
        del buf14
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.select]
        buf16 = torch.ops.aten.select.int(buf15, 3, 1)
        buf17 = buf16
        # Topologically Sorted Source Nodes: [getitem_1], Original ATen: [aten.select]
        buf18 = torch.ops.aten.select.int(buf9, 3, 1)
        buf19 = buf18
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.copy]
        buf20 = torch.ops.aten.copy.default(buf17, buf19)
        del buf16
        del buf17
        del buf18
        del buf19
        del buf9
        buf21 = buf20
        del buf20
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf22 = torch.ops.aten.select_scatter.default(buf15, buf21, 3, 1)
        del buf15
        del buf21
        buf23 = buf22
        del buf22
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.unsqueeze]
        buf24 = torch.ops.aten.unsqueeze.default(buf23, 4)
        buf25 = buf24
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.permute]
        buf26 = torch.ops.aten.permute.default(buf25, [0, 1, 3, 4, 2])
        buf27 = buf26
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.permute]
        buf28 = torch.ops.aten.permute.default(buf27, [0, 1, 2, 4, 3])
        buf29 = buf28
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.view]
        buf30 = torch.ops.aten.reshape.default(buf29, [16, 2, 4])
        buf31 = buf30
        # Topologically Sorted Source Nodes: [xk_ft_], Original ATen: [aten.zeros]
        buf32 = torch.ops.aten.full.default([4, 4, 4, 2], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf33 = buf32
        del buf32
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.select]
        buf34 = torch.ops.aten.select.int(buf33, 3, 0)
        buf35 = buf34
        # Topologically Sorted Source Nodes: [xk_ft], Original ATen: [aten._fft_r2c]
        buf36 = torch.ops.aten._fft_r2c.default(reinterpret_tensor(arg1_1, (4, 4, 4, 4), (64, 1, 4, 16), 0), [3], 0, True)
        del arg1_1
        buf37 = buf36
        del buf36
        # Topologically Sorted Source Nodes: [getitem_2], Original ATen: [aten.select]
        buf38 = torch.ops.aten.select.int(buf37, 3, 0)
        buf39 = buf38
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.copy]
        buf40 = torch.ops.aten.copy.default(buf35, buf39)
        del buf34
        del buf35
        del buf38
        del buf39
        buf41 = buf40
        del buf40
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf42 = torch.ops.aten.select_scatter.default(buf33, buf41, 3, 0)
        del buf33
        del buf41
        buf43 = buf42
        del buf42
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.select]
        buf44 = torch.ops.aten.select.int(buf43, 3, 1)
        buf45 = buf44
        # Topologically Sorted Source Nodes: [getitem_3], Original ATen: [aten.select]
        buf46 = torch.ops.aten.select.int(buf37, 3, 1)
        buf47 = buf46
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.copy]
        buf48 = torch.ops.aten.copy.default(buf45, buf47)
        del buf37
        del buf44
        del buf45
        del buf46
        del buf47
        buf49 = buf48
        del buf48
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf50 = torch.ops.aten.select_scatter.default(buf43, buf49, 3, 1)
        del buf43
        del buf49
        buf51 = buf50
        del buf50
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.unsqueeze]
        buf52 = torch.ops.aten.unsqueeze.default(buf51, 4)
        buf53 = buf52
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.permute]
        buf54 = torch.ops.aten.permute.default(buf53, [0, 1, 4, 3, 2])
        buf55 = buf54
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.permute]
        buf56 = torch.ops.aten.permute.default(buf55, [0, 1, 4, 3, 2])
        buf57 = buf56
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.view]
        buf58 = torch.ops.aten.reshape.default(buf57, [16, 4, 2])
        buf59 = buf58
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.bmm]
        buf60 = torch.ops.aten.bmm.default(buf31, buf59)
        del buf23
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        del buf30
        del buf31
        del buf52
        del buf53
        del buf54
        del buf55
        del buf56
        del buf57
        del buf58
        del buf59
        buf61 = buf60
        del buf60
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.view]
        buf62 = torch.ops.aten.reshape.default(buf61, [4, 4, 2, 1, 2])
        buf63 = buf62
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.permute]
        buf64 = torch.ops.aten.permute.default(buf63, [0, 1, 2, 4, 3])
        buf65 = buf64
        # Topologically Sorted Source Nodes: [xqk_ft], Original ATen: [aten.view]
        buf66 = torch.ops.aten.reshape.default(buf65, [4, 4, 2, 2])
        buf67 = buf66
        # Topologically Sorted Source Nodes: [xqk_ft_1], Original ATen: [aten.tanh]
        buf68 = torch.ops.aten.tanh.default(buf67)
        del buf61
        del buf62
        del buf63
        del buf64
        del buf65
        del buf66
        del buf67
        buf69 = buf68
        del buf68
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.unsqueeze]
        buf70 = torch.ops.aten.unsqueeze.default(buf69, 4)
        buf71 = buf70
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.permute]
        buf72 = torch.ops.aten.permute.default(buf71, [0, 1, 4, 2, 3])
        buf73 = buf72
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.permute]
        buf74 = torch.ops.aten.permute.default(buf73, [0, 1, 3, 4, 2])
        buf75 = buf74
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.view]
        buf76 = torch.ops.aten.reshape.default(buf75, [16, 2, 2])
        buf77 = buf76
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.unsqueeze]
        buf78 = torch.ops.aten.unsqueeze.default(buf51, 4)
        buf79 = buf78
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.permute]
        buf80 = torch.ops.aten.permute.default(buf79, [0, 1, 2, 4, 3])
        buf81 = buf80
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.permute]
        buf82 = torch.ops.aten.permute.default(buf81, [0, 1, 4, 2, 3])
        buf83 = buf82
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.view]
        buf84 = torch.ops.aten.reshape.default(buf83, [16, 2, 4])
        buf85 = buf84
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.bmm]
        buf86 = torch.ops.aten.bmm.default(buf77, buf85)
        del buf51
        del buf69
        del buf70
        del buf71
        del buf72
        del buf73
        del buf74
        del buf75
        del buf76
        del buf77
        del buf78
        del buf79
        del buf80
        del buf81
        del buf82
        del buf83
        del buf84
        del buf85
        buf87 = buf86
        del buf86
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.view]
        buf88 = torch.ops.aten.reshape.default(buf87, [4, 4, 2, 1, 4])
        buf89 = buf88
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.permute]
        buf90 = torch.ops.aten.permute.default(buf89, [0, 1, 4, 2, 3])
        buf91 = buf90
        # Topologically Sorted Source Nodes: [xqkv_ft], Original ATen: [aten.view]
        buf92 = torch.ops.aten.reshape.default(buf91, [4, 4, 4, 2])
        buf93 = buf92
        # Topologically Sorted Source Nodes: [getitem_4], Original ATen: [aten.select]
        buf94 = torch.ops.aten.select.int(buf93, 3, 0)
        buf95 = buf94
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.copy]
        buf96 = torch.ops.aten.copy.default(buf3, buf95)
        del buf2
        del buf3
        del buf94
        del buf95
        buf97 = buf96
        del buf96
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf98 = torch.ops.aten.select_scatter.default(buf1, buf97, 3, 0)
        del buf1
        del buf97
        buf99 = buf98
        del buf98
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.select]
        buf100 = torch.ops.aten.select.int(buf99, 3, 1)
        buf101 = buf100
        # Topologically Sorted Source Nodes: [getitem_5], Original ATen: [aten.select]
        buf102 = torch.ops.aten.select.int(buf93, 3, 1)
        buf103 = buf102
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.copy]
        buf104 = torch.ops.aten.copy.default(buf101, buf103)
        del buf100
        del buf101
        del buf102
        del buf103
        del buf87
        del buf88
        del buf89
        del buf90
        del buf91
        del buf92
        del buf93
        buf105 = buf104
        del buf104
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf106 = torch.ops.aten.select_scatter.default(buf99, buf105, 3, 1)
        del buf105
        del buf99
        buf107 = buf106
        del buf106
        # Topologically Sorted Source Nodes: [truediv], Original ATen: [aten.div]
        buf108 = torch.ops.aten.div.Scalar(buf107, 4)
        del buf107
        buf109 = buf108
        del buf108
        # Topologically Sorted Source Nodes: [truediv_1], Original ATen: [aten.div]
        buf110 = torch.ops.aten.div.Scalar(buf109, 4)
        del buf109
        buf111 = buf110
        del buf110
        # Topologically Sorted Source Nodes: [fft_irfft], Original ATen: [aten._fft_c2r]
        buf112 = torch.ops.aten._fft_c2r.default(buf111, [3], 2, 4)
        del buf111
        buf113 = buf112
        del buf112
    return (reinterpret_tensor(buf113, (4, 4, 4, 4), (64, 1, 4, 16), 0), )


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
