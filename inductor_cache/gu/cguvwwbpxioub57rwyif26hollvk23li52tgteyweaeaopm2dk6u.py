# AOT ID: ['41_inference']
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


# kernel path: inductor_cache/gl/cgllenxjy6zyejnutee7ccicmjyes5bjhk35vvvahwejntlpzi2t.py
# Topologically Sorted Source Nodes: [embeddings], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   embeddings => div
# Graph fragment:
#   %div : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %expand), kwargs = {})
triton_poi_fused_div_0 = async_compile.triton('triton_poi_fused_div_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bh/cbhbc6j2awi5qli2o6miq2pssaq2hckuaezep2fiiuil5ejas6if.py
# Topologically Sorted Source Nodes: [add, mul, dist], Original ATen: [aten.add, aten.mul, aten.sub]
# Source node to ATen node mapping:
#   add => add
#   dist => sub
#   mul => mul
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 2.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %mul), kwargs = {})
triton_poi_fused_add_mul_sub_1 = async_compile.triton('triton_poi_fused_add_mul_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sub_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp10 + tmp21
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 - tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fd/cfdnj5pcrquxho4e2sntiyle4cmmflsxbnht7zzcptfa5z5yy2hi.py
# Topologically Sorted Source Nodes: [and_, skip_mask, skip_mask_1, equal_labels_mask, float_1, mul_1, marign_mask, clamp_min_, sub_2, mul_2, add_1, relu, loss, sum_3, sum_4, add_2, truediv], Original ATen: [aten.bitwise_and, aten.bitwise_not, aten._to_copy, aten.eq, aten.mul, aten.sub, aten.clamp_min, aten.add, aten.relu, aten.sum, aten.div]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   and_ => bitwise_and
#   clamp_min_ => clamp_min_1
#   equal_labels_mask => eq
#   float_1 => convert_element_type
#   loss => mul_3
#   marign_mask => sub_1
#   mul_1 => mul_1
#   mul_2 => mul_2
#   relu => relu
#   skip_mask => bitwise_not
#   skip_mask_1 => convert_element_type_1
#   sub_2 => sub_2
#   sum_3 => sum_4
#   sum_4 => sum_5
#   truediv => div_1
# Graph fragment:
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%unsqueeze_5, %unsqueeze_6), kwargs = {})
#   %bitwise_not : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_and,), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_not, torch.float32), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, 1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, 1.0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %sub_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 0.2), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %relu), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_3,), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%convert_element_type_1,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_5, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_4, %add_2), kwargs = {})
triton_per_fused__to_copy_add_bitwise_and_bitwise_not_clamp_min_div_eq_mul_relu_sub_sum_2 = async_compile.triton('triton_per_fused__to_copy_add_bitwise_and_bitwise_not_clamp_min_div_eq_mul_relu_sub_sum_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bitwise_and_bitwise_not_clamp_min_div_eq_mul_relu_sub_sum_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_bitwise_and_bitwise_not_clamp_min_div_eq_mul_relu_sub_sum_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex // 4
    r4 = (rindex % 16)
    r0 = (rindex % 4)
    r2 = rindex // 16
    tmp0 = tl.load(in_ptr0 + (r3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (r4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (r0 + 4*r2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (r4), None, eviction_policy='evict_last')
    tmp1 = -1.0
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = (tmp3 != 0)
    tmp6 = tmp5 == tmp1
    tmp7 = tmp6.to(tl.int64)
    tmp8 = (tmp7 != 0)
    tmp9 = tmp4 & tmp8
    tmp10 = tmp9 == 0
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp5 == tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 2.0
    tmp16 = tmp14 * tmp15
    tmp17 = 1.0
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tmp21 - tmp17
    tmp23 = tmp18 * tmp22
    tmp24 = 0.2
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tmp11 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp32 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp34 = tl.sum(tmp32, 1)[:, None]
    tmp35 = 1e-08
    tmp36 = tmp34 + tmp35
    tmp37 = tmp31 / tmp36
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp37, None)
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
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embeddings], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_0.run(arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(buf0, (4, 4), (1, 4), 0), out=buf1)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [add, mul, dist], Original ATen: [aten.add, aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sub_1.run(buf2, buf0, 16, grid=grid(16), stream=stream0)
        del buf0
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [and_, skip_mask, skip_mask_1, equal_labels_mask, float_1, mul_1, marign_mask, clamp_min_, sub_2, mul_2, add_1, relu, loss, sum_3, sum_4, add_2, truediv], Original ATen: [aten.bitwise_and, aten.bitwise_not, aten._to_copy, aten.eq, aten.mul, aten.sub, aten.clamp_min, aten.add, aten.relu, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_bitwise_and_bitwise_not_clamp_min_div_eq_mul_relu_sub_sum_2.run(buf5, arg1_1, buf2, 1, 64, grid=grid(1), stream=stream0)
        del arg1_1
        del buf2
    return (buf5, )


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
