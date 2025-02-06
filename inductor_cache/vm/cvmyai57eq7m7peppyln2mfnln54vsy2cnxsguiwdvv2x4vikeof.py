# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/jm/cjm72rv33mrffs5zm5lcnhosa35dh2b6qlhwoncr7dmr4soy6l32.py
# Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, mul_1, sum_1], Original ATen: [aten.index, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   getitem => index
#   getitem_1 => index_1
#   getitem_2 => index_2
#   mul => mul
#   mul_1 => mul_1
#   sum_1 => sum_1
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_1, [%primals_2]), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_3, [%primals_4]), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %index_1), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_1, [%primals_5]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %index_2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
triton_poi_fused_index_mul_sum_0 = async_compile.triton('triton_poi_fused_index_mul_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp16 = tl.load(in_ptr4 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp25 = tl.load(in_ptr0 + (1))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp35 = tl.load(in_ptr4 + (1))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp45 = tl.load(in_ptr0 + (2))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp55 = tl.load(in_ptr4 + (2))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp65 = tl.load(in_ptr0 + (3))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp75 = tl.load(in_ptr4 + (3))
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK])
    tmp2 = tl.full([XBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 4), "index out of bounds: 0 <= tmp5 < 4")
    tmp7 = tl.load(in_ptr1 + (tmp5), None, eviction_policy='evict_last')
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp9 + tmp2
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert(((0 <= tmp12) & (tmp12 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp12 < 4")
    tmp14 = tl.load(in_ptr3 + (4*tmp12), xmask, eviction_policy='evict_last')
    tmp15 = tmp8 * tmp14
    tmp18 = tmp17 + tmp2
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert((0 <= tmp20) & (tmp20 < 4), "index out of bounds: 0 <= tmp20 < 4")
    tmp22 = tl.load(in_ptr1 + (tmp20), None, eviction_policy='evict_last')
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp15 * tmp23
    tmp27 = tmp26 + tmp2
    tmp28 = tmp26 < 0
    tmp29 = tl.where(tmp28, tmp27, tmp26)
    tl.device_assert((0 <= tmp29) & (tmp29 < 4), "index out of bounds: 0 <= tmp29 < 4")
    tmp31 = tl.load(in_ptr1 + (tmp29), None, eviction_policy='evict_last')
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.load(in_ptr3 + (1 + 4*tmp12), xmask, eviction_policy='evict_last')
    tmp34 = tmp32 * tmp33
    tmp37 = tmp36 + tmp2
    tmp38 = tmp36 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp36)
    tl.device_assert((0 <= tmp39) & (tmp39 < 4), "index out of bounds: 0 <= tmp39 < 4")
    tmp41 = tl.load(in_ptr1 + (tmp39), None, eviction_policy='evict_last')
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp34 * tmp42
    tmp44 = tmp24 + tmp43
    tmp47 = tmp46 + tmp2
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert((0 <= tmp49) & (tmp49 < 4), "index out of bounds: 0 <= tmp49 < 4")
    tmp51 = tl.load(in_ptr1 + (tmp49), None, eviction_policy='evict_last')
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tl.load(in_ptr3 + (2 + 4*tmp12), xmask, eviction_policy='evict_last')
    tmp54 = tmp52 * tmp53
    tmp57 = tmp56 + tmp2
    tmp58 = tmp56 < 0
    tmp59 = tl.where(tmp58, tmp57, tmp56)
    tl.device_assert((0 <= tmp59) & (tmp59 < 4), "index out of bounds: 0 <= tmp59 < 4")
    tmp61 = tl.load(in_ptr1 + (tmp59), None, eviction_policy='evict_last')
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp54 * tmp62
    tmp64 = tmp44 + tmp63
    tmp67 = tmp66 + tmp2
    tmp68 = tmp66 < 0
    tmp69 = tl.where(tmp68, tmp67, tmp66)
    tl.device_assert((0 <= tmp69) & (tmp69 < 4), "index out of bounds: 0 <= tmp69 < 4")
    tmp71 = tl.load(in_ptr1 + (tmp69), None, eviction_policy='evict_last')
    tmp72 = tmp71.to(tl.float32)
    tmp73 = tl.load(in_ptr3 + (3 + 4*tmp12), xmask, eviction_policy='evict_last')
    tmp74 = tmp72 * tmp73
    tmp77 = tmp76 + tmp2
    tmp78 = tmp76 < 0
    tmp79 = tl.where(tmp78, tmp77, tmp76)
    tl.device_assert((0 <= tmp79) & (tmp79 < 4), "index out of bounds: 0 <= tmp79 < 4")
    tmp81 = tl.load(in_ptr1 + (tmp79), None, eviction_policy='evict_last')
    tmp82 = tmp81.to(tl.float32)
    tmp83 = tmp74 * tmp82
    tmp84 = tmp64 + tmp83
    tl.store(out_ptr0 + (x0), tmp84, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, ), (1, ))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, getitem_1, mul, getitem_2, mul_1, sum_1], Original ATen: [aten.index, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_mul_sum_0.run(primals_2, primals_1, primals_4, primals_3, primals_5, buf0, 4, grid=grid(4), stream=stream0)
        del primals_3
    return (buf0, primals_1, primals_2, primals_4, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
