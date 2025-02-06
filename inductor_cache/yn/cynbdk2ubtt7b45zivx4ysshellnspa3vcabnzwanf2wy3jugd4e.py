# AOT ID: ['4_inference']
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


# kernel path: inductor_cache/3u/c3uxxyibwsonzcciwtj4hq3nsbvjr22efxpjmt2erccnk6yiiqme.py
# Topologically Sorted Source Nodes: [mask, ge, lt, mask_1], Original ATen: [aten.arange, aten.ge, aten.lt, aten.bitwise_and]
# Source node to ATen node mapping:
#   ge => ge
#   lt => lt
#   mask => add_1, convert_element_type_3, iota, mul_2
#   mask_1 => bitwise_and
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 0), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%convert_element_type_3, %squeeze), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%convert_element_type_3, %squeeze_1), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %lt), kwargs = {})
triton_poi_fused_arange_bitwise_and_ge_lt_0 = async_compile.triton('triton_poi_fused_arange_bitwise_and_ge_lt_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_bitwise_and_ge_lt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_ge_lt_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 4.0
    tmp7 = tmp6 - tmp5
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = x0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 >= tmp10
    tmp14 = tmp5.to(tl.int64)
    tmp15 = tmp9 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp12 < tmp16
    tmp18 = tmp13 & tmp17
    tl.store(out_ptr0 + (x0), tmp18, xmask)
''', device_str='cuda')


cpp_fused_ge_sub_1 = async_compile.cpp_pybinding(['const float*', 'const float*', 'bool*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = in_ptr1[static_cast<int64_t>(0L)];
                auto tmp2 = static_cast<float>(2.0);
                auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                auto tmp4 = static_cast<float>(4.0);
                auto tmp5 = decltype(tmp4)(tmp4 - tmp3);
                auto tmp6 = decltype(tmp0)(tmp0 * tmp5);
                auto tmp7 = c10::convert<int64_t>(tmp6);
                auto tmp8 = c10::convert<int64_t>(tmp3);
                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                auto tmp10 = decltype(tmp9)(tmp9 - tmp7);
                auto tmp11 = static_cast<int64_t>(2);
                auto tmp12 = tmp10 >= tmp11;
                out_ptr0[static_cast<int64_t>(0L)] = tmp12;
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
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    # Topologically Sorted Source Nodes: [rand_1], Original ATen: [aten.rand]
    buf0 = torch.ops.aten.rand.default([1], device=device(type='cpu'), pin_memory=False)
    buf1 = buf0
    del buf0
    # Topologically Sorted Source Nodes: [rand], Original ATen: [aten.rand]
    buf2 = torch.ops.aten.rand.default([1], device=device(type='cpu'), pin_memory=False)
    buf3 = buf2
    del buf2
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((4, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [mask, ge, lt, mask_1], Original ATen: [aten.arange, aten.ge, aten.lt, aten.bitwise_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_bitwise_and_ge_lt_0.run(buf1, buf3, buf4, 4, grid=grid(4), stream=stream0)
    buf5 = empty_strided_cpu((), (), torch.bool)
    cpp_fused_ge_sub_1(buf1, buf3, buf5)
    return (buf5, reinterpret_tensor(arg0_1, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf4, (4, 1), (1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
