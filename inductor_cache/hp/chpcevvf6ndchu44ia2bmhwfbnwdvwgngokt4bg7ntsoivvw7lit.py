# AOT ID: ['14_inference']
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


# kernel path: inductor_cache/ug/cug6ujacudwpebahhr2zdb6nodecjd23zqlxzvvaa2fzk55ocyyx.py
# Topologically Sorted Source Nodes: [pow_2, sum_2, feat_t_norm, add_1, feat_t, setitem_1], Original ATen: [aten.pow, aten.sum, aten.sqrt, aten.add, aten.div, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   add_1 => add_1
#   feat_t => div_1
#   feat_t_norm => sqrt_1
#   pow_2 => pow_2
#   setitem_1 => full_default_1, index_put_1
#   sum_2 => sum_2
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [1], True), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_1, 1e-06), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg1_1, %add_1), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_1 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%div_1, [%ne_1], %full_default_1), kwargs = {})
triton_poi_fused_add_div_index_put_lift_fresh_pow_sqrt_sum_0 = async_compile.triton('triton_poi_fused_add_div_index_put_lift_fresh_pow_sqrt_sum_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_index_put_lift_fresh_pow_sqrt_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_index_put_lift_fresh_pow_sqrt_sum_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = tmp0 / tmp14
    tmp16 = tmp15 != tmp15
    tmp17 = 0.0
    tmp18 = tl.where(tmp16, tmp17, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vp/cvp5pwcgmvaiq2tthqlm3wvavxodrb7hbluyfz3jomul5y4hzukd.py
# Topologically Sorted Source Nodes: [add_3, feat_t_cos_sim_1, sum_4, feat_t_cond_prob, add_4, add_2, feat_s_cos_sim_1, sum_3, feat_s_cond_prob, add_5, truediv_6, log, mul, loss], Original ATen: [aten.add, aten.div, aten.sum, aten.log, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   feat_s_cond_prob => div_4
#   feat_s_cos_sim_1 => div_2
#   feat_t_cond_prob => div_5
#   feat_t_cos_sim_1 => div_3
#   log => log
#   loss => mean
#   mul => mul
#   sum_3 => sum_3
#   sum_4 => sum_4
#   truediv_6 => div_6
# Graph fragment:
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, 1.0), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_3, 2.0), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_3, [1], True), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_3, %sum_4), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_5, 1e-06), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, 1.0), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_2, 2.0), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_2, [1], True), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_2, %sum_3), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, 1e-06), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, %add_5), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div_6,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %log), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%mul,), kwargs = {})
triton_per_fused_add_div_log_mean_mul_sum_1 = async_compile.triton('triton_per_fused_add_div_log_mean_mul_sum_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log_mean_mul_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log_mean_mul_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    r1 = rindex // 4
    tmp0 = tl.load(in_ptr0 + (r2), None)
    tmp5 = tl.load(in_ptr0 + (4*r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (r2), None)
    tmp26 = tl.load(in_ptr1 + (4*r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr1 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp6 = tmp5 + tmp1
    tmp7 = tmp6 * tmp3
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 * tmp3
    tmp11 = tmp7 + tmp10
    tmp13 = tmp12 + tmp1
    tmp14 = tmp13 * tmp3
    tmp15 = tmp11 + tmp14
    tmp17 = tmp16 + tmp1
    tmp18 = tmp17 * tmp3
    tmp19 = tmp15 + tmp18
    tmp20 = tmp4 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp27 = tmp26 + tmp1
    tmp28 = tmp27 * tmp3
    tmp30 = tmp29 + tmp1
    tmp31 = tmp30 * tmp3
    tmp32 = tmp28 + tmp31
    tmp34 = tmp33 + tmp1
    tmp35 = tmp34 * tmp3
    tmp36 = tmp32 + tmp35
    tmp38 = tmp37 + tmp1
    tmp39 = tmp38 * tmp3
    tmp40 = tmp36 + tmp39
    tmp41 = tmp25 / tmp40
    tmp42 = tmp41 + tmp21
    tmp43 = tmp22 / tmp42
    tmp44 = tl_math.log(tmp43)
    tmp45 = tmp20 * tmp44
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.sum(tmp46, 1)[:, None]
    tmp49 = 16.0
    tmp50 = tmp48 / tmp49
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp50, None)
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
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [pow_2, sum_2, feat_t_norm, add_1, feat_t, setitem_1], Original ATen: [aten.pow, aten.sum, aten.sqrt, aten.add, aten.div, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_index_put_lift_fresh_pow_sqrt_sum_0.run(buf1, arg1_1, 16, grid=grid(16), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [feat_t_cos_sim], Original ATen: [aten.mm]
        extern_kernels.mm(buf1, reinterpret_tensor(buf1, (4, 4), (1, 4), 0), out=buf2)
        buf4 = buf1; del buf1  # reuse
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [pow_1, sum_1, feat_s_norm, add, feat_s, setitem], Original ATen: [aten.pow, aten.sum, aten.sqrt, aten.add, aten.div, aten.lift_fresh, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_index_put_lift_fresh_pow_sqrt_sum_0.run(buf5, arg0_1, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [feat_s_cos_sim], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, reinterpret_tensor(buf5, (4, 4), (1, 4), 0), out=buf6)
        del buf5
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [add_3, feat_t_cos_sim_1, sum_4, feat_t_cond_prob, add_4, add_2, feat_s_cos_sim_1, sum_3, feat_s_cond_prob, add_5, truediv_6, log, mul, loss], Original ATen: [aten.add, aten.div, aten.sum, aten.log, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_log_mean_mul_sum_1.run(buf8, buf2, buf6, 1, 16, grid=grid(1), stream=stream0)
        del buf2
        del buf6
    return (buf8, )


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
