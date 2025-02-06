# AOT ID: ['30_inference']
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


# kernel path: inductor_cache/ci/cci3f6rtr7rlyx6eglpw6xquc5nomjcz3fe3xaxbaafbu6ecsih2.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x => getitem
# Graph fragment:
#   %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 3)
    x0 = (xindex % 3)
    x2 = xindex // 9
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=float("-inf"))
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + x0 + 4*x1 + 16*x2), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + x0 + 4*x1 + 16*x2), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2 + x0
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + ((-2) + x0 + 4*x1 + 16*x2), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = x1
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp36 & tmp9
    tmp38 = tl.load(in_ptr0 + ((-1) + x0 + 4*x1 + 16*x2), tmp37 & xmask, other=float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp32)
    tmp40 = tmp36 & tmp15
    tmp41 = tl.load(in_ptr0 + (x0 + 4*x1 + 16*x2), tmp40 & xmask, other=float("-inf"))
    tmp42 = triton_helpers.maximum(tmp41, tmp39)
    tmp43 = tmp36 & tmp22
    tmp44 = tl.load(in_ptr0 + (1 + x0 + 4*x1 + 16*x2), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp42)
    tmp46 = tmp36 & tmp29
    tmp47 = tl.load(in_ptr0 + (2 + x0 + 4*x1 + 16*x2), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = 1 + x1
    tmp50 = tmp49 >= tmp1
    tmp51 = tmp49 < tmp3
    tmp52 = tmp50 & tmp51
    tmp53 = tmp52 & tmp9
    tmp54 = tl.load(in_ptr0 + (3 + x0 + 4*x1 + 16*x2), tmp53 & xmask, other=float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp48)
    tmp56 = tmp52 & tmp15
    tmp57 = tl.load(in_ptr0 + (4 + x0 + 4*x1 + 16*x2), tmp56 & xmask, other=float("-inf"))
    tmp58 = triton_helpers.maximum(tmp57, tmp55)
    tmp59 = tmp52 & tmp22
    tmp60 = tl.load(in_ptr0 + (5 + x0 + 4*x1 + 16*x2), tmp59 & xmask, other=float("-inf"))
    tmp61 = triton_helpers.maximum(tmp60, tmp58)
    tmp62 = tmp52 & tmp29
    tmp63 = tl.load(in_ptr0 + (6 + x0 + 4*x1 + 16*x2), tmp62 & xmask, other=float("-inf"))
    tmp64 = triton_helpers.maximum(tmp63, tmp61)
    tmp65 = 2 + x1
    tmp66 = tmp65 >= tmp1
    tmp67 = tmp65 < tmp3
    tmp68 = tmp66 & tmp67
    tmp69 = tmp68 & tmp9
    tmp70 = tl.load(in_ptr0 + (7 + x0 + 4*x1 + 16*x2), tmp69 & xmask, other=float("-inf"))
    tmp71 = triton_helpers.maximum(tmp70, tmp64)
    tmp72 = tmp68 & tmp15
    tmp73 = tl.load(in_ptr0 + (8 + x0 + 4*x1 + 16*x2), tmp72 & xmask, other=float("-inf"))
    tmp74 = triton_helpers.maximum(tmp73, tmp71)
    tmp75 = tmp68 & tmp22
    tmp76 = tl.load(in_ptr0 + (9 + x0 + 4*x1 + 16*x2), tmp75 & xmask, other=float("-inf"))
    tmp77 = triton_helpers.maximum(tmp76, tmp74)
    tmp78 = tmp68 & tmp29
    tmp79 = tl.load(in_ptr0 + (10 + x0 + 4*x1 + 16*x2), tmp78 & xmask, other=float("-inf"))
    tmp80 = triton_helpers.maximum(tmp79, tmp77)
    tl.store(out_ptr0 + (x4), tmp80, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0.run(arg0_1, buf0, 144, grid=grid(144), stream=stream0)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
