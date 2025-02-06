# AOT ID: ['18_inference']
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


cpp_fused_div_lift_fresh_mul_sum_0 = async_compile.cpp_pybinding(['float*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* out_ptr0,
                       float* out_ptr1)
{
    {
        {
            float tmp_acc0 = 0;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = x0;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = static_cast<int64_t>(2);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<int64_t>(1);
                            auto tmp5 = tmp1 < tmp4;
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = static_cast<float>(3.0);
                            auto tmp8 = tmp5 ? tmp6 : tmp7;
                            auto tmp9 = static_cast<int64_t>(3);
                            auto tmp10 = tmp1 < tmp9;
                            auto tmp11 = tmp10 ? tmp7 : tmp6;
                            auto tmp12 = tmp3 ? tmp8 : tmp11;
                            auto tmp13 = x1;
                            auto tmp14 = c10::convert<int64_t>(tmp13);
                            auto tmp15 = tmp14 < tmp2;
                            auto tmp16 = tmp14 < tmp4;
                            auto tmp17 = tmp16 ? tmp6 : tmp7;
                            auto tmp18 = tmp14 < tmp9;
                            auto tmp19 = tmp18 ? tmp7 : tmp6;
                            auto tmp20 = tmp15 ? tmp17 : tmp19;
                            auto tmp21 = decltype(tmp12)(tmp12 * tmp20);
                            tmp_acc0 = tmp_acc0 + tmp21;
                        }
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp22 = out_ptr0[static_cast<int64_t>(0L)];
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(2);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(1);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = static_cast<float>(3.0);
                        auto tmp8 = tmp5 ? tmp6 : tmp7;
                        auto tmp9 = static_cast<int64_t>(3);
                        auto tmp10 = tmp1 < tmp9;
                        auto tmp11 = tmp10 ? tmp7 : tmp6;
                        auto tmp12 = tmp3 ? tmp8 : tmp11;
                        auto tmp13 = x1;
                        auto tmp14 = c10::convert<int64_t>(tmp13);
                        auto tmp15 = tmp14 < tmp2;
                        auto tmp16 = tmp14 < tmp4;
                        auto tmp17 = tmp16 ? tmp6 : tmp7;
                        auto tmp18 = tmp14 < tmp9;
                        auto tmp19 = tmp18 ? tmp7 : tmp6;
                        auto tmp20 = tmp15 ? tmp17 : tmp19;
                        auto tmp21 = decltype(tmp12)(tmp12 * tmp20);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = decltype(tmp23)(tmp23 * tmp6);
                        out_ptr1[static_cast<int64_t>(x1 + 4L*x0)] = tmp24;
                    }
                }
            }
        }
    }
}
''')


# kernel path: inductor_cache/eh/ceh7eipyu5doksnoinj6yhkh67qp5uc7z53oos7m7ypxm2xkcfbc.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   out_3 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_3, [0, 0, 1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_1 = async_compile.triton('triton_poi_fused_constant_pad_nd_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
    tl.store(out_ptr0 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ph/cphowovclkmcmozxd73u2farwajkpj2ggxvytajljqcjrxc32dsg.py
# Topologically Sorted Source Nodes: [flip], Original ATen: [aten.flip]
# Source node to ATen node mapping:
#   flip => rev
# Graph fragment:
#   %rev : [num_users=1] = call_function[target=torch.ops.prims.rev.default](args = (%device_put, [0, 1]), kwargs = {})
triton_poi_fused_flip_2 = async_compile.triton('triton_poi_fused_flip_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_flip_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_flip_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hy/chyrubhc65c4u6s6kcid52rsdl4qc4cswiofjtxolycjpujcaje7.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   output => view_7
# Graph fragment:
#   %view_7 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_5, [-1, 4, 2, 2]), kwargs = {})
triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 6*x1 + 9*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    buf0 = empty_strided_cpu((), (), torch.float32)
    buf1 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    cpp_fused_div_lift_fresh_mul_sum_0(buf0, buf1)
    del buf0
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf2.copy_(buf1, False)
        del buf1
        buf3 = empty_strided_cuda((16, 6, 6, 1), (36, 6, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_1.run(arg0_1, buf3, 576, grid=grid(576), stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [flip], Original ATen: [aten.flip]
        stream0 = get_raw_stream(0)
        triton_poi_fused_flip_2.run(buf2, buf4, 16, grid=grid(16), stream=stream0)
        del buf2
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(reinterpret_tensor(buf3, (16, 1, 6, 6), (36, 0, 6, 1), 0), reinterpret_tensor(buf4, (1, 1, 4, 4), (0, 0, 4, 1), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (16, 1, 3, 3), (9, 9, 3, 1))
        del buf3
        del buf4
        buf6 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf5, buf6, 64, grid=grid(64), stream=stream0)
        del buf5
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
