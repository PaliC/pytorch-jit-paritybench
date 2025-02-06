# AOT ID: ['6_inference']
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


# kernel path: inductor_cache/qv/cqv2rmkefyux2zup3uqriwm2qp22cp56wx6gmridv4u7wsqy37sj.py
# Topologically Sorted Source Nodes: [stft_1, stft_2, stft_3, stft_4], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   stft_1 => mul
#   stft_2 => mul_1
#   stft_3 => mul_2
#   stft_4 => mul_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unfold, %hann_window), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unfold_1, %hann_window), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unfold_2, %hann_window), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unfold_3, %hann_window), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp8 = tmp7 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr1 + (x2), tmp4, xmask)
    tl.store(out_ptr2 + (x2), tmp6, xmask)
    tl.store(out_ptr3 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4), (16, 4, 1))
    # Topologically Sorted Source Nodes: [stft], Original ATen: [aten.zeros]
    buf0 = torch.ops.aten.full.default([4, 3, 1, 4], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cpu'), pin_memory=False)
    buf1 = buf0
    del buf0
    # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.select]
    buf2 = torch.ops.aten.select.int(buf1, 3, 0)
    buf3 = buf2
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [window], Original ATen: [aten.hann_window]
        buf4 = torch.ops.aten.hann_window.default(4, device=device(type='cuda', index=0), pin_memory=False)
        buf5 = buf4
        del buf4
        buf6 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        buf28 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        buf39 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stft_1, stft_2, stft_3, stft_4], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, buf5, buf6, buf17, buf28, buf39, 16, grid=grid(16), stream=stream0)
        del arg0_1
        del buf5
        # Topologically Sorted Source Nodes: [stft_1], Original ATen: [aten.mul, aten._fft_r2c]
        buf7 = torch.ops.aten._fft_r2c.default(buf6, [2], 0, True)
        del buf6
        buf8 = buf7
        del buf7
        # Topologically Sorted Source Nodes: [stft_1], Original ATen: [aten.transpose]
        buf9 = torch.ops.aten.permute.default(buf8, [0, 2, 1])
        buf10 = buf9
    # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.copy]
    buf11 = torch.ops.aten.copy.default(buf3, buf10)
    del buf10
    del buf2
    del buf3
    del buf8
    del buf9
    buf12 = buf11
    del buf11
    # Topologically Sorted Source Nodes: [], Original ATen: []
    buf13 = torch.ops.aten.select_scatter.default(buf1, buf12, 3, 0)
    del buf1
    del buf12
    buf14 = buf13
    del buf13
    # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.select]
    buf15 = torch.ops.aten.select.int(buf14, 3, 1)
    buf16 = buf15
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [stft_2], Original ATen: [aten.mul, aten._fft_r2c]
        buf18 = torch.ops.aten._fft_r2c.default(buf17, [2], 0, True)
        del buf17
        buf19 = buf18
        del buf18
        # Topologically Sorted Source Nodes: [stft_2], Original ATen: [aten.transpose]
        buf20 = torch.ops.aten.permute.default(buf19, [0, 2, 1])
        buf21 = buf20
    # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.copy]
    buf22 = torch.ops.aten.copy.default(buf16, buf21)
    del buf15
    del buf16
    del buf19
    del buf20
    del buf21
    buf23 = buf22
    del buf22
    # Topologically Sorted Source Nodes: [], Original ATen: []
    buf24 = torch.ops.aten.select_scatter.default(buf14, buf23, 3, 1)
    del buf14
    del buf23
    buf25 = buf24
    del buf24
    # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.select]
    buf26 = torch.ops.aten.select.int(buf25, 3, 2)
    buf27 = buf26
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [stft_3], Original ATen: [aten.mul, aten._fft_r2c]
        buf29 = torch.ops.aten._fft_r2c.default(buf28, [2], 0, True)
        del buf28
        buf30 = buf29
        del buf29
        # Topologically Sorted Source Nodes: [stft_3], Original ATen: [aten.transpose]
        buf31 = torch.ops.aten.permute.default(buf30, [0, 2, 1])
        buf32 = buf31
    # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.copy]
    buf33 = torch.ops.aten.copy.default(buf27, buf32)
    del buf26
    del buf27
    del buf30
    del buf31
    del buf32
    buf34 = buf33
    del buf33
    # Topologically Sorted Source Nodes: [], Original ATen: []
    buf35 = torch.ops.aten.select_scatter.default(buf25, buf34, 3, 2)
    del buf25
    del buf34
    buf36 = buf35
    del buf35
    # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.select]
    buf37 = torch.ops.aten.select.int(buf36, 3, 3)
    buf38 = buf37
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [stft_4], Original ATen: [aten.mul, aten._fft_r2c]
        buf40 = torch.ops.aten._fft_r2c.default(buf39, [2], 0, True)
        del buf39
        buf41 = buf40
        del buf40
        # Topologically Sorted Source Nodes: [stft_4], Original ATen: [aten.transpose]
        buf42 = torch.ops.aten.permute.default(buf41, [0, 2, 1])
        buf43 = buf42
    # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.copy]
    buf44 = torch.ops.aten.copy.default(buf38, buf43)
    del buf37
    del buf38
    del buf41
    del buf42
    del buf43
    buf45 = buf44
    del buf44
    # Topologically Sorted Source Nodes: [], Original ATen: []
    buf46 = torch.ops.aten.select_scatter.default(buf36, buf45, 3, 3)
    del buf36
    del buf45
    buf47 = buf46
    return (buf47, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
