# AOT ID: ['17_forward']
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


# kernel path: inductor_cache/6y/c6yubw5jdeeyukawefzi6acatll6fdi2kbwxvc2i4dychjyvvyh2.py
# Topologically Sorted Source Nodes: [sign, positive_part, neg, negative_part, weight_diff_1, weight_diff_2, weight_diff_3, sub, weight], Original ATen: [aten.sign, aten.relu, aten.neg, aten.add, aten.mul, aten.tanh, aten.rsub]
# Source node to ATen node mapping:
#   neg => neg
#   negative_part => relu_1
#   positive_part => relu
#   sign => sign
#   sub => sub
#   weight => mul_1
#   weight_diff_1 => add
#   weight_diff_2 => mul
#   weight_diff_3 => tanh
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%primals_1,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%view_1,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%neg,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %relu_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %add), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %tanh), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 0.5), kwargs = {})
triton_poi_fused_add_mul_neg_relu_rsub_sign_tanh_0 = async_compile.triton('triton_poi_fused_add_mul_neg_relu_rsub_sign_tanh_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_neg_relu_rsub_sign_tanh_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_neg_relu_rsub_sign_tanh_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 < tmp0
    tmp3 = tmp2.to(tl.int8)
    tmp4 = tmp0 < tmp1
    tmp5 = tmp4.to(tl.int8)
    tmp6 = tmp3 - tmp5
    tmp7 = tmp6.to(tmp0.dtype)
    tmp9 = triton_helpers.maximum(tmp1, tmp8)
    tmp10 = -tmp8
    tmp11 = triton_helpers.maximum(tmp1, tmp10)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp7 * tmp12
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (x0), tmp18, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_diff], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_3, reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf0)
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sign, positive_part, neg, negative_part, weight_diff_1, weight_diff_2, weight_diff_3, sub, weight], Original ATen: [aten.sign, aten.relu, aten.neg, aten.add, aten.mul, aten.tanh, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_neg_relu_rsub_sign_tanh_0.run(primals_1, buf0, buf1, 256, grid=grid(256), stream=stream0)
    return (buf1, primals_1, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
