# AOT ID: ['11_inference']
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


# kernel path: inductor_cache/dw/cdwrcjjnxuk2gycct5remiksa4qtmbsz6kak42u6bzjstsxwgowl.py
# Topologically Sorted Source Nodes: [truediv, mul, sin, mul_1, cos, mul_2, sin_1, mul_3, cos_1, mul_4, sin_2, mul_5, cos_2, mul_6, sin_3, mul_7, cos_3], Original ATen: [aten.div, aten.mul, aten.sin, aten.cos]
# Source node to ATen node mapping:
#   cos => cos
#   cos_1 => cos_1
#   cos_2 => cos_2
#   cos_3 => cos_3
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   sin => sin
#   sin_1 => sin_1
#   sin_2 => sin_2
#   sin_3 => sin_3
#   truediv => div
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, 1.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1.0), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1.0), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 2.5198421478271484), kwargs = {})
#   %sin_1 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 2.5198421478271484), kwargs = {})
#   %cos_1 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_3,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 6.349603652954102), kwargs = {})
#   %sin_2 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 6.349603652954102), kwargs = {})
#   %cos_2 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_5,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 16.0), kwargs = {})
#   %sin_3 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 16.0), kwargs = {})
#   %cos_3 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_7,), kwargs = {})
triton_poi_fused_cos_div_mul_sin_0 = async_compile.triton('triton_poi_fused_cos_div_mul_sin_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 5, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cos_div_mul_sin_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cos_div_mul_sin_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl_math.sin(tmp2)
    tmp4 = tl_math.cos(tmp2)
    tmp5 = 2.5198421478271484
    tmp6 = tmp0 * tmp5
    tmp7 = tl_math.sin(tmp6)
    tmp8 = tl_math.cos(tmp6)
    tmp9 = 6.349603652954102
    tmp10 = tmp0 * tmp9
    tmp11 = tl_math.sin(tmp10)
    tmp12 = tl_math.cos(tmp10)
    tmp13 = 16.0
    tmp14 = tmp0 * tmp13
    tmp15 = tl_math.sin(tmp14)
    tmp16 = tl_math.cos(tmp14)
    tl.store(out_ptr0 + (x0 + 36*x1), tmp2, xmask)
    tl.store(out_ptr1 + (x0 + 36*x1), tmp3, xmask)
    tl.store(out_ptr2 + (x0 + 36*x1), tmp4, xmask)
    tl.store(out_ptr3 + (x0 + 36*x1), tmp7, xmask)
    tl.store(out_ptr4 + (x0 + 36*x1), tmp8, xmask)
    tl.store(out_ptr5 + (x0 + 36*x1), tmp11, xmask)
    tl.store(out_ptr6 + (x0 + 36*x1), tmp12, xmask)
    tl.store(out_ptr7 + (x0 + 36*x1), tmp15, xmask)
    tl.store(out_ptr8 + (x0 + 36*x1), tmp16, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf9 = empty_strided_cuda((4, 4, 4, 36), (576, 144, 36, 1), torch.float32)
        buf0 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 0)  # alias
        buf1 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 4)  # alias
        buf2 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 8)  # alias
        buf3 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 12)  # alias
        buf4 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 16)  # alias
        buf5 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 20)  # alias
        buf6 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 24)  # alias
        buf7 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 28)  # alias
        buf8 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 32)  # alias
        # Topologically Sorted Source Nodes: [truediv, mul, sin, mul_1, cos, mul_2, sin_1, mul_3, cos_1, mul_4, sin_2, mul_5, cos_2, mul_6, sin_3, mul_7, cos_3], Original ATen: [aten.div, aten.mul, aten.sin, aten.cos]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cos_div_mul_sin_0.run(arg0_1, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, 256, grid=grid(256), stream=stream0)
        del arg0_1
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
