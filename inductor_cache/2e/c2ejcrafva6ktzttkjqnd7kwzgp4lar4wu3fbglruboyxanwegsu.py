# AOT ID: ['65_inference']
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


# kernel path: inductor_cache/zm/czmpcckzs74wqcdeo64nirgwz65nsfvrvxi7jmuxgotnyg5qsoqa.py
# Topologically Sorted Source Nodes: [diag, negative_similarity, hard_negative_ids], Original ATen: [aten.diag_embed, aten.add, aten.argmax]
# Source node to ATen node mapping:
#   diag => eq, full_default, full_default_1, iota_1, where
#   hard_negative_ids => argmax
#   negative_similarity => add
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%iota_1, %unsqueeze_3), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 4], -1000000000), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %full_default_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %where), kwargs = {})
#   %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%add, -1), kwargs = {})
triton_poi_fused_add_argmax_diag_embed_0 = async_compile.triton('triton_poi_fused_add_argmax_diag_embed_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_argmax_diag_embed_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_argmax_diag_embed_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], -1000000000, tl.int64)
    tmp5 = tl.where(tmp3, tmp4, tmp1)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 + tmp6
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp9 == tmp2
    tmp11 = tl.where(tmp10, tmp4, tmp1)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp8 + tmp12
    tmp14 = tmp7 > tmp13
    tmp15 = tmp7 == tmp13
    tmp16 = tmp7 != tmp7
    tmp17 = tmp13 != tmp13
    tmp18 = tmp16 > tmp17
    tmp19 = tmp14 | tmp18
    tmp20 = tmp16 & tmp17
    tmp21 = tmp15 | tmp20
    tmp22 = tmp1 < tmp9
    tmp23 = tmp21 & tmp22
    tmp24 = tmp19 | tmp23
    tmp25 = tl.where(tmp24, tmp7, tmp13)
    tmp26 = tl.where(tmp24, tmp1, tmp9)
    tmp28 = tl.full([1], 2, tl.int64)
    tmp29 = tmp28 == tmp2
    tmp30 = tl.where(tmp29, tmp4, tmp1)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp27 + tmp31
    tmp33 = tmp25 > tmp32
    tmp34 = tmp25 == tmp32
    tmp35 = tmp25 != tmp25
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tmp26 < tmp28
    tmp42 = tmp40 & tmp41
    tmp43 = tmp38 | tmp42
    tmp44 = tl.where(tmp43, tmp25, tmp32)
    tmp45 = tl.where(tmp43, tmp26, tmp28)
    tmp47 = tl.full([1], 3, tl.int64)
    tmp48 = tmp47 == tmp2
    tmp49 = tl.where(tmp48, tmp4, tmp1)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp46 + tmp50
    tmp52 = tmp44 > tmp51
    tmp53 = tmp44 == tmp51
    tmp54 = tmp44 != tmp44
    tmp55 = tmp51 != tmp51
    tmp56 = tmp54 > tmp55
    tmp57 = tmp52 | tmp56
    tmp58 = tmp54 & tmp55
    tmp59 = tmp53 | tmp58
    tmp60 = tmp45 < tmp47
    tmp61 = tmp59 & tmp60
    tmp62 = tmp57 | tmp61
    tmp63 = tl.where(tmp62, tmp44, tmp51)
    tmp64 = tl.where(tmp62, tmp45, tmp47)
    tl.store(out_ptr0 + (x0), tmp64, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/54/c5452zcps7efe7b7sip7zkfazlzvoctfab4rg3p3vjkjg3rdpqtl.py
# Topologically Sorted Source Nodes: [positive_similarities, sub, negative_similarities, add_1, loss, sum_1, loss_1], Original ATen: [aten.index, aten.rsub, aten.add, aten.relu, aten.sum, aten.div]
# Source node to ATen node mapping:
#   add_1 => add_1
#   loss => relu
#   loss_1 => div
#   negative_similarities => index
#   positive_similarities => index_1
#   sub => sub
#   sum_1 => sum_1
# Graph fragment:
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_3, [%iota, %iota]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.3, %index_1), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_3, [%iota, %argmax]), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %index), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%relu,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, 4), kwargs = {})
triton_poi_fused_add_div_index_relu_rsub_sum_1 = async_compile.triton('triton_poi_fused_add_div_index_relu_rsub_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_index_relu_rsub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_index_relu_rsub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (5))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr1 + (1))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (10))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp31 = tl.load(in_ptr1 + (2))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (15))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + (3))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp2 = 0.3
    tmp3 = tmp2 - tmp1
    tmp6 = tl.full([XBLOCK], 4, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert((0 <= tmp9) & (tmp9 < 4), "index out of bounds: 0 <= tmp9 < 4")
    tmp11 = tl.load(in_ptr0 + (tmp9), None, eviction_policy='evict_last')
    tmp12 = tmp3 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp17 = tmp2 - tmp16
    tmp20 = tmp19 + tmp6
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tl.device_assert((0 <= tmp22) & (tmp22 < 4), "index out of bounds: 0 <= tmp22 < 4")
    tmp24 = tl.load(in_ptr0 + (4 + tmp22), None, eviction_policy='evict_last')
    tmp25 = tmp17 + tmp24
    tmp26 = triton_helpers.maximum(tmp13, tmp25)
    tmp27 = tmp14 + tmp26
    tmp30 = tmp2 - tmp29
    tmp33 = tmp32 + tmp6
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 4), "index out of bounds: 0 <= tmp35 < 4")
    tmp37 = tl.load(in_ptr0 + (8 + tmp35), None, eviction_policy='evict_last')
    tmp38 = tmp30 + tmp37
    tmp39 = triton_helpers.maximum(tmp13, tmp38)
    tmp40 = tmp27 + tmp39
    tmp43 = tmp2 - tmp42
    tmp46 = tmp45 + tmp6
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tl.device_assert((0 <= tmp48) & (tmp48 < 4), "index out of bounds: 0 <= tmp48 < 4")
    tmp50 = tl.load(in_ptr0 + (12 + tmp48), None, eviction_policy='evict_last')
    tmp51 = tmp43 + tmp50
    tmp52 = triton_helpers.maximum(tmp13, tmp51)
    tmp53 = tmp40 + tmp52
    tmp54 = 0.25
    tmp55 = tmp53 * tmp54
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp55, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pairwise_similarity], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(arg1_1, (1, 4, 4), (0, 1, 4), 0), out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [diag, negative_similarity, hard_negative_ids], Original ATen: [aten.diag_embed, aten.add, aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_argmax_diag_embed_0.run(buf0, buf1, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [positive_similarities, sub, negative_similarities, add_1, loss, sum_1, loss_1], Original ATen: [aten.index, aten.rsub, aten.add, aten.relu, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_index_relu_rsub_sum_1.run(buf3, buf0, buf1, 1, grid=grid(1), stream=stream0)
        del buf0
        del buf1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
