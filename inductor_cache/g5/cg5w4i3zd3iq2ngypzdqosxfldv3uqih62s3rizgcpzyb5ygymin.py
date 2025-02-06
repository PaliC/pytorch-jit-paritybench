# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/24/c24jvlhd54cansafjpv2vq4noyzsorv73ev6cqcmtia6gbr4keij.py
# Topologically Sorted Source Nodes: [out, mul_1, out_1, mul_2, out_2, mul_3, out_3, mul_4, out_4, mul_5, out_5], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   out => mul
#   out_1 => add
#   out_2 => add_1
#   out_3 => add_2
#   out_4 => add_3
#   out_5 => add_4
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %select_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %select_11), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_5), kwargs = {})
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr0 + (1))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (64 + x0), xmask)
    tmp9 = tl.load(in_ptr0 + (2))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (128 + x0), xmask)
    tmp14 = tl.load(in_ptr0 + (3))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + (192 + x0), xmask)
    tmp19 = tl.load(in_ptr0 + (4))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp21 = tl.load(in_ptr1 + (256 + x0), xmask)
    tmp24 = tl.load(in_ptr0 + (5))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (320 + x0), xmask)
    tmp3 = tmp1 * tmp2
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 + tmp7
    tmp12 = tmp10 * tmp11
    tmp13 = tmp8 + tmp12
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 + tmp17
    tmp22 = tmp20 * tmp21
    tmp23 = tmp18 + tmp22
    tmp27 = tmp25 * tmp26
    tmp28 = tmp23 + tmp27
    tl.store(in_out_ptr0 + (x0), tmp28, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (6, ), (1, ))
    assert_size_stride(primals_2, (6, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [out, mul_1, out_1, mul_2, out_2, mul_3, out_3, mul_4, out_4, mul_5, out_5], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_0.run(buf1, primals_1, primals_2, 64, grid=grid(64), stream=stream0)
        del primals_1
    return (buf1, primals_2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((6, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
