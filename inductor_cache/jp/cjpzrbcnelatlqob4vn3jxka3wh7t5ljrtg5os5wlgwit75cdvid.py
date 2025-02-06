# AOT ID: ['8_forward']
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


# kernel path: inductor_cache/x5/cx54gyrtyz5zrqfp2yli543mnyhxjfzssjghysqku2bnyqvcxfyh.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   hx => full_default
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_0 = async_compile.triton('triton_poi_fused_zeros_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2x/c2xycwthzwxhqx4ucinp7qianckf2rsjlrwffnzejzixdopufylz.py
# Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   rnn_out => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3],), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 4, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (16, 4), (4, 1))
    assert_size_stride(primals_3, (16, 4), (4, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_0.run(buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (4, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 16), (1, 4), 0), out=buf1)
        buf2 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (4, 16), (1, 4), 0), out=buf2)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten._thnn_fused_lstm_cell]
        buf3 = torch.ops.aten._thnn_fused_lstm_cell.default(buf1, buf2, buf0, primals_4, primals_5)
        buf4 = buf3[0]
        buf5 = buf3[1]
        buf6 = buf3[2]
        del buf3
        buf7 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (4, 4), (4, 1), 16), reinterpret_tensor(primals_2, (4, 16), (1, 4), 0), out=buf7)
        buf8 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_3, (4, 16), (1, 4), 0), out=buf8)
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten._thnn_fused_lstm_cell]
        buf9 = torch.ops.aten._thnn_fused_lstm_cell.default(buf7, buf8, buf5, primals_4, primals_5)
        buf10 = buf9[0]
        buf11 = buf9[1]
        buf12 = buf9[2]
        del buf9
        buf13 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (4, 4), (4, 1), 32), reinterpret_tensor(primals_2, (4, 16), (1, 4), 0), out=buf13)
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf10, reinterpret_tensor(primals_3, (4, 16), (1, 4), 0), out=buf14)
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten._thnn_fused_lstm_cell]
        buf15 = torch.ops.aten._thnn_fused_lstm_cell.default(buf13, buf14, buf11, primals_4, primals_5)
        buf16 = buf15[0]
        buf17 = buf15[1]
        buf18 = buf15[2]
        del buf15
        buf19 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (4, 4), (4, 1), 48), reinterpret_tensor(primals_2, (4, 16), (1, 4), 0), out=buf19)
        del primals_2
        buf20 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_3, (4, 16), (1, 4), 0), out=buf20)
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten._thnn_fused_lstm_cell]
        buf21 = torch.ops.aten._thnn_fused_lstm_cell.default(buf19, buf20, buf17, primals_4, primals_5)
        del buf19
        del primals_4
        del primals_5
        buf22 = buf21[0]
        buf23 = buf21[1]
        buf24 = buf21[2]
        del buf21
        buf25 = reinterpret_tensor(buf20, (4, 4, 4), (16, 4, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf4, buf10, buf16, buf22, buf25, 64, grid=grid(64), stream=stream0)
        del buf22
    return (buf25, buf0, reinterpret_tensor(primals_1, (4, 4), (4, 1), 0), buf4, buf5, buf6, reinterpret_tensor(primals_1, (4, 4), (4, 1), 16), buf10, buf11, buf12, reinterpret_tensor(primals_1, (4, 4), (4, 1), 32), buf16, buf17, buf18, reinterpret_tensor(primals_1, (4, 4), (4, 1), 48), buf23, buf24, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
