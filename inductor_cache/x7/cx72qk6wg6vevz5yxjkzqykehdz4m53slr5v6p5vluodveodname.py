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


# kernel path: inductor_cache/py/cpyl4dxkliuqciypkznnovq47k7h7sd4qjw6ysrthrwdzm7zpawe.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_6 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.float64), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp6.to(tl.float64)
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7o/c7owzoyhqlce7sa5lj6wnsmawjrk4jbk5fjtfs3qtvpzd3t2ased.py
# Topologically Sorted Source Nodes: [diag_embed], Original ATen: [aten.diag_embed]
# Source node to ATen node mapping:
#   diag_embed => full_default, where
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%view_6, %permute_2, %full_default), kwargs = {})
triton_poi_fused_diag_embed_1 = async_compile.triton('triton_poi_fused_diag_embed_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_diag_embed_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_diag_embed_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 == tmp1
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = tl.full([1], 0.0, tl.float64)
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wh/cwhgslwsd3dweh64rslexwldu6wqgo2cpnycguwck6kz5vcawkbd.py
# Topologically Sorted Source Nodes: [x_10, squeeze], Original ATen: [aten.linalg_vector_norm, aten.div, aten.squeeze]
# Source node to ATen node mapping:
#   squeeze => squeeze
#   x_10 => div, pow_2, sum_1
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_14, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_14, %expand_6), kwargs = {})
#   %squeeze : [num_users=1] = call_function[target=torch.ops.aten.squeeze.default](args = (%div,), kwargs = {})
triton_per_fused_div_linalg_vector_norm_squeeze_2 = async_compile.triton('triton_per_fused_div_linalg_vector_norm_squeeze_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_linalg_vector_norm_squeeze_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_linalg_vector_norm_squeeze_2(in_out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1e-12, tl.float64)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp0 / tmp8
    tl.store(in_out_ptr0 + (r1 + 16*x0), tmp9, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (64, 4, 1), (4, 1, 1), 0), reinterpret_tensor(arg0_1, (64, 1, 4), (4, 1, 1), 0), out=buf0)
        del arg0_1
        buf1 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float64)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(buf0, buf1, 256, grid=grid(256), stream=stream0)
        del buf0
        # Topologically Sorted Source Nodes: [x_6, svd], Original ATen: [aten._to_copy, aten._linalg_svd]
        buf2 = torch.ops.aten._linalg_svd.default(buf1)
        buf3 = buf2[0]
        buf4 = buf2[1]
        buf5 = buf2[2]
        del buf2
        buf6 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [diag_embed], Original ATen: [aten.diag_embed]
        stream0 = get_raw_stream(0)
        triton_poi_fused_diag_embed_1.run(buf4, buf6, 256, grid=grid(256), stream=stream0)
        del buf4
        buf7 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float64)
        # Topologically Sorted Source Nodes: [diag_embed, matmul_1], Original ATen: [aten.diag_embed, aten.bmm]
        extern_kernels.bmm(buf3, buf6, out=buf7)
        del buf3
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf7, buf5, out=buf8)
        del buf5
        del buf7
        buf10 = reinterpret_tensor(buf8, (4, 4, 16), (64, 16, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_10, squeeze], Original ATen: [aten.linalg_vector_norm, aten.div, aten.squeeze]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_linalg_vector_norm_squeeze_2.run(buf10, 16, 16, grid=grid(16), stream=stream0)
    return (buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
