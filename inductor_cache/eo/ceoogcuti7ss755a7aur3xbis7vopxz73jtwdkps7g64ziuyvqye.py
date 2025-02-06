# AOT ID: ['19_inference']
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


# kernel path: inductor_cache/3j/c3jxtxpoxxemhwhaw66tzdyw3o2wpex6c3yecdcjd43hmi4j35hg.py
# Topologically Sorted Source Nodes: [mse_loss, mean, losses], Original ATen: [aten.mse_loss, aten.mean]
# Source node to ATen node mapping:
#   losses => mean_1
#   mean => mean
#   mse_loss => pow_1, sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg0_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean, [1]), kwargs = {})
triton_poi_fused_mean_mse_loss_0 = async_compile.triton('triton_poi_fused_mean_mse_loss_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mse_loss_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mse_loss_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp4 = tl.load(in_ptr0 + (4 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr1 + (4 + x0 + 64*x1), xmask)
    tmp9 = tl.load(in_ptr0 + (8 + x0 + 64*x1), xmask)
    tmp10 = tl.load(in_ptr1 + (8 + x0 + 64*x1), xmask)
    tmp14 = tl.load(in_ptr0 + (12 + x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr1 + (12 + x0 + 64*x1), xmask)
    tmp21 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp22 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp25 = tl.load(in_ptr0 + (20 + x0 + 64*x1), xmask)
    tmp26 = tl.load(in_ptr1 + (20 + x0 + 64*x1), xmask)
    tmp30 = tl.load(in_ptr0 + (24 + x0 + 64*x1), xmask)
    tmp31 = tl.load(in_ptr1 + (24 + x0 + 64*x1), xmask)
    tmp35 = tl.load(in_ptr0 + (28 + x0 + 64*x1), xmask)
    tmp36 = tl.load(in_ptr1 + (28 + x0 + 64*x1), xmask)
    tmp42 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp43 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp46 = tl.load(in_ptr0 + (36 + x0 + 64*x1), xmask)
    tmp47 = tl.load(in_ptr1 + (36 + x0 + 64*x1), xmask)
    tmp51 = tl.load(in_ptr0 + (40 + x0 + 64*x1), xmask)
    tmp52 = tl.load(in_ptr1 + (40 + x0 + 64*x1), xmask)
    tmp56 = tl.load(in_ptr0 + (44 + x0 + 64*x1), xmask)
    tmp57 = tl.load(in_ptr1 + (44 + x0 + 64*x1), xmask)
    tmp63 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp64 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp67 = tl.load(in_ptr0 + (52 + x0 + 64*x1), xmask)
    tmp68 = tl.load(in_ptr1 + (52 + x0 + 64*x1), xmask)
    tmp72 = tl.load(in_ptr0 + (56 + x0 + 64*x1), xmask)
    tmp73 = tl.load(in_ptr1 + (56 + x0 + 64*x1), xmask)
    tmp77 = tl.load(in_ptr0 + (60 + x0 + 64*x1), xmask)
    tmp78 = tl.load(in_ptr1 + (60 + x0 + 64*x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = 4.0
    tmp20 = tmp18 / tmp19
    tmp23 = tmp21 - tmp22
    tmp24 = tmp23 * tmp23
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tmp24 + tmp28
    tmp32 = tmp30 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tmp29 + tmp33
    tmp37 = tmp35 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tmp34 + tmp38
    tmp40 = tmp39 / tmp19
    tmp41 = tmp20 + tmp40
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp44
    tmp48 = tmp46 - tmp47
    tmp49 = tmp48 * tmp48
    tmp50 = tmp45 + tmp49
    tmp53 = tmp51 - tmp52
    tmp54 = tmp53 * tmp53
    tmp55 = tmp50 + tmp54
    tmp58 = tmp56 - tmp57
    tmp59 = tmp58 * tmp58
    tmp60 = tmp55 + tmp59
    tmp61 = tmp60 / tmp19
    tmp62 = tmp41 + tmp61
    tmp65 = tmp63 - tmp64
    tmp66 = tmp65 * tmp65
    tmp69 = tmp67 - tmp68
    tmp70 = tmp69 * tmp69
    tmp71 = tmp66 + tmp70
    tmp74 = tmp72 - tmp73
    tmp75 = tmp74 * tmp74
    tmp76 = tmp71 + tmp75
    tmp79 = tmp77 - tmp78
    tmp80 = tmp79 * tmp79
    tmp81 = tmp76 + tmp80
    tmp82 = tmp81 / tmp19
    tmp83 = tmp62 + tmp82
    tmp84 = tmp83 / tmp19
    tl.store(out_ptr0 + (x2), tmp84, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mse_loss, mean, losses], Original ATen: [aten.mse_loss, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mse_loss_0.run(arg1_1, arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        del arg1_1
        # Topologically Sorted Source Nodes: [topk], Original ATen: [aten.topk]
        buf1 = torch.ops.aten.topk.default(buf0, 4)
        del buf0
        buf2 = buf1[0]
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
