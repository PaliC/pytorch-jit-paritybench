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


# kernel path: inductor_cache/w3/cw3krlil6navzcnlnt5zyvivozsiiq7ijlstk6uwprvutxpik2fr.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sqrt, %sqrt_1, %sqrt_2, %sqrt_3], 1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 4)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.load(in_ptr1 + (x0 + 16*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 + tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 2, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + (x0 + 16*x2), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 * tmp19
    tmp21 = tl.load(in_ptr3 + (x0 + 16*x2), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 + tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.sqrt(tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp18, tmp26, tmp27)
    tmp29 = tmp0 >= tmp16
    tmp30 = tl.full([1], 3, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tl.load(in_ptr4 + (x0 + 16*x2), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp33 * tmp33
    tmp35 = tl.load(in_ptr5 + (x0 + 16*x2), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp32, tmp40, tmp41)
    tmp43 = tmp0 >= tmp30
    tmp44 = tl.full([1], 4, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tl.load(in_ptr6 + (x0 + 16*x2), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp46 * tmp46
    tmp48 = tl.load(in_ptr7 + (x0 + 16*x2), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 * tmp48
    tmp50 = tmp47 + tmp49
    tmp51 = 1e-06
    tmp52 = tmp50 + tmp51
    tmp53 = libdevice.sqrt(tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp43, tmp53, tmp54)
    tmp56 = tl.where(tmp32, tmp42, tmp55)
    tmp57 = tl.where(tmp18, tmp28, tmp56)
    tmp58 = tl.where(tmp4, tmp14, tmp57)
    tl.store(out_ptr0 + (x3), tmp58, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg2_1, (1, 1, 3, 3), (9, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_i_v], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 0), arg1_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_i_h], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 0), arg2_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_i_v_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 16), arg1_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_i_h_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 16), arg2_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_i_v_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 32), arg1_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_i_h_2], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 32), arg2_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_i_v_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 48), arg1_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 1, 4, 4), (16, 16, 4, 1))
        del arg1_1
        # Topologically Sorted Source Nodes: [x_i_h_3], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(arg0_1, (4, 1, 4, 4), (64, 0, 4, 1), 48), arg2_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 1, 4, 4), (16, 16, 4, 1))
        del arg0_1
        del arg2_1
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, 256, grid=grid(256), stream=stream0)
        del buf0
        del buf1
        del buf2
        del buf3
        del buf4
        del buf5
        del buf6
        del buf7
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
