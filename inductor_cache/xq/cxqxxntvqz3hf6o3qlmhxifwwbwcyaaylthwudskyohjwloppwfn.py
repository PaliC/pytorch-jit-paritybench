# AOT ID: ['109_inference']
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


# kernel path: inductor_cache/aw/cawudshzg2bdlnknayrcmsoeoqqgdyhxiarrddzoqcjdyhrgl5zm.py
# Topologically Sorted Source Nodes: [logits, logs], Original ATen: [aten._to_copy, aten._log_softmax]
# Source node to ATen node mapping:
#   logits => convert_element_type
#   logs => amax, sub
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type, [1], True), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %amax), kwargs = {})
triton_poi_fused__log_softmax__to_copy_0 = async_compile.triton('triton_poi_fused__log_softmax__to_copy_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = triton_helpers.maximum(tmp3, tmp5)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = triton_helpers.maximum(tmp9, tmp11)
    tmp13 = tmp1 - tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfxv2mcqauthrxytgtwmwwo6fx2sly473z2blbibzkaanmmjpn26.py
# Topologically Sorted Source Nodes: [logs, setitem, scatter_, mul, sum_2, loss, setitem_1], Original ATen: [aten._log_softmax, aten.lift_fresh, aten.index_put, aten.scatter, aten.mul, aten.sum, aten.neg]
# Source node to ATen node mapping:
#   logs => exp, log, sub_1, sum_2
#   loss => neg
#   mul => mul
#   scatter_ => scatter_upon_const_tensor
#   setitem => full_default, index_put
#   setitem_1 => full_default_2, index_put_1
#   sum_2 => sum_3
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %log), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%arg1_1, [%eq], %full_default), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [4, 4], background_val: 0.025, dtype: torch.float32, dim: 1, selector: %unsqueeze_1, val: 0.9})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %scatter_upon_const_tensor), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_3,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%neg, [%eq], %full_default_2), kwargs = {})
triton_poi_fused__log_softmax_index_put_lift_fresh_mul_neg_scatter_sum_1 = async_compile.triton('triton_poi_fused__log_softmax_index_put_lift_fresh_mul_neg_scatter_sum_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_index_put_lift_fresh_mul_neg_scatter_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_index_put_lift_fresh_mul_neg_scatter_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp6 = tl_math.exp(tmp5)
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tmp6 + tmp8
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tmp9 + tmp11
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp12 + tmp14
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp5 - tmp16
    tmp18 = tmp4 == tmp3
    tmp19 = 0.9
    tmp20 = 0.025
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp7 - tmp16
    tmp24 = tl.full([1], 1, tl.int64)
    tmp25 = tmp4 == tmp24
    tmp26 = tl.where(tmp25, tmp19, tmp20)
    tmp27 = tmp23 * tmp26
    tmp28 = tmp22 + tmp27
    tmp29 = tmp10 - tmp16
    tmp30 = tl.full([1], 2, tl.int64)
    tmp31 = tmp4 == tmp30
    tmp32 = tl.where(tmp31, tmp19, tmp20)
    tmp33 = tmp29 * tmp32
    tmp34 = tmp28 + tmp33
    tmp35 = tmp13 - tmp16
    tmp36 = tl.full([1], 3, tl.int64)
    tmp37 = tmp4 == tmp36
    tmp38 = tl.where(tmp37, tmp19, tmp20)
    tmp39 = tmp35 * tmp38
    tmp40 = tmp34 + tmp39
    tmp41 = -tmp40
    tmp42 = 0.0
    tmp43 = tl.where(tmp2, tmp42, tmp41)
    tl.store(in_out_ptr0 + (x0), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zr/czrpihhyt3rvwsxciv4yhz4eif2g2omtq65f4hiet7cspubtrloa.py
# Topologically Sorted Source Nodes: [ignore, sum_3, eq_1, n_valid, loss_1], Original ATen: [aten.eq, aten.sum, aten.div]
# Source node to ATen node mapping:
#   eq_1 => eq_1
#   ignore => eq
#   loss_1 => div
#   n_valid => sum_1
#   sum_3 => sum_4
# Graph fragment:
#   %eq : [num_users=3] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, -100), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%index_put_1,), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%eq, 0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%eq_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_4, %sum_1), kwargs = {})
triton_poi_fused_div_eq_sum_2 = async_compile.triton('triton_poi_fused_div_eq_sum_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_eq_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_eq_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (3))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (1))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (2))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp33 = tl.load(in_ptr1 + (3))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp10 = tmp7 + tmp9
    tmp13 = tl.full([1], -100, tl.int64)
    tmp14 = tmp12 == tmp13
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 == tmp16
    tmp18 = tmp17.to(tl.int64)
    tmp21 = tmp20 == tmp13
    tmp22 = tmp21.to(tl.int64)
    tmp23 = tmp22 == tmp16
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tmp18 + tmp24
    tmp28 = tmp27 == tmp13
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp29 == tmp16
    tmp31 = tmp30.to(tl.int64)
    tmp32 = tmp25 + tmp31
    tmp35 = tmp34 == tmp13
    tmp36 = tmp35.to(tl.int64)
    tmp37 = tmp36 == tmp16
    tmp38 = tmp37.to(tl.int64)
    tmp39 = tmp32 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp10 / tmp40
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp41, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits, logs], Original ATen: [aten._to_copy, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__to_copy_0.run(arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [logs, setitem, scatter_, mul, sum_2, loss, setitem_1], Original ATen: [aten._log_softmax, aten.lift_fresh, aten.index_put, aten.scatter, aten.mul, aten.sum, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_index_put_lift_fresh_mul_neg_scatter_sum_1.run(buf3, arg1_1, buf0, 4, grid=grid(4), stream=stream0)
        del buf0
        buf4 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [ignore, sum_3, eq_1, n_valid, loss_1], Original ATen: [aten.eq, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_eq_sum_2.run(buf3, arg1_1, buf4, 1, grid=grid(1), stream=stream0)
        del arg1_1
        del buf3
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
