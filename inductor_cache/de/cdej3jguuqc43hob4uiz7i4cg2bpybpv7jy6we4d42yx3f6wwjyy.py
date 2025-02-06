# AOT ID: ['4_inference']
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


# kernel path: inductor_cache/az/cazq5m6vqq4vldqpzepr5b7cr4btbt4rwgikrclf5wamkaclpm5g.py
# Topologically Sorted Source Nodes: [tgt_spans], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   tgt_spans => cat
# Graph fragment:
#   %cat : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%select, %select_1, %select_2, %select_3],), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (16 + x0 + 4*((-4) + x1)), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr0 + (32 + x0 + 4*((-8) + x1)), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 16, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (48 + x0 + 4*((-12) + x1)), tmp16 & xmask, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eo/ceotr6ywccnluilnldvb7ecuepnf4hhl4vk2jrrkrgbc2rorobwl.py
# Topologically Sorted Source Nodes: [mul_4, areas2, add_2, right, left, sub_4, inter, union, iou, right_1, left_1, sub_6, enclosing_area, sub_7, truediv_1, sub_8, cost_giou, mul_5, C], Original ATen: [aten.mul, aten.sub, aten.add, aten.minimum, aten.maximum, aten.clamp, aten.div, aten.neg]
# Source node to ATen node mapping:
#   C => add_3
#   add_2 => add_2
#   areas2 => sub_3
#   cost_giou => neg
#   enclosing_area => clamp_min_1
#   inter => clamp_min
#   iou => div
#   left => maximum
#   left_1 => minimum_1
#   mul_4 => mul_4
#   mul_5 => mul_5
#   right => minimum
#   right_1 => maximum_1
#   sub_4 => sub_4
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   truediv_1 => div_1
#   union => sub_5
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_cdist_forward, 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_14, %select_15), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_6, %sub_3), kwargs = {})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%select_18, %select_19), kwargs = {})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%select_16, %select_17), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %maximum), kwargs = {})
#   %clamp_min : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_4, 0), kwargs = {})
#   %sub_5 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %clamp_min), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_min, %sub_5), kwargs = {})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%select_22, %select_23), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%select_20, %select_21), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%maximum_1, %minimum_1), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_6, 0), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %sub_5), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_7, %clamp_min_1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %div_1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sub_8,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, 1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
triton_poi_fused_add_clamp_div_maximum_minimum_mul_neg_sub_1 = async_compile.triton('triton_poi_fused_add_clamp_div_maximum_minimum_mul_neg_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_maximum_minimum_mul_neg_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_maximum_minimum_mul_neg_sub_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp76 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = tl.full([1], 1, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp0 < tmp0
    tmp4 = tl.load(in_ptr0 + (4*x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 - tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp3, tmp8, tmp9)
    tmp11 = tmp0 >= tmp0
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (4*x1), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr0 + (1 + 4*x1), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp11, tmp18, tmp19)
    tmp21 = tl.where(tmp3, tmp10, tmp20)
    tmp22 = tmp1 >= tmp1
    tmp23 = tmp1 < tmp0
    tmp24 = tl.load(in_ptr0 + (4*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr0 + (1 + 4*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = 0.5
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 - tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp23, tmp28, tmp29)
    tmp31 = tmp1 >= tmp0
    tmp32 = tmp1 < tmp12
    tmp33 = tl.load(in_ptr0 + (4*x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr0 + (1 + 4*x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = 0.5
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp31, tmp37, tmp38)
    tmp40 = tl.where(tmp23, tmp30, tmp39)
    tmp41 = tmp21 - tmp40
    tmp42 = tl.load(in_ptr1 + (4*x0), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr1 + (1 + 4*x0), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp43 * tmp6
    tmp45 = tmp42 - tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp3, tmp45, tmp46)
    tmp48 = tl.load(in_ptr1 + (4*x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.load(in_ptr1 + (1 + 4*x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 * tmp16
    tmp51 = tmp48 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp11, tmp51, tmp52)
    tmp54 = tl.where(tmp3, tmp47, tmp53)
    tmp55 = tl.load(in_ptr1 + (4*x0), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr1 + (1 + 4*x0), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp56 * tmp26
    tmp58 = tmp55 - tmp57
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp23, tmp58, tmp59)
    tmp61 = tl.load(in_ptr1 + (4*x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr1 + (1 + 4*x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tmp62 * tmp35
    tmp64 = tmp61 + tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp31, tmp64, tmp65)
    tmp67 = tl.where(tmp23, tmp60, tmp66)
    tmp68 = tmp54 - tmp67
    tmp69 = tmp41 + tmp68
    tmp70 = triton_helpers.minimum(tmp21, tmp54)
    tmp71 = triton_helpers.maximum(tmp40, tmp67)
    tmp72 = tmp70 - tmp71
    tmp73 = triton_helpers.maximum(tmp21, tmp54)
    tmp74 = triton_helpers.minimum(tmp40, tmp67)
    tmp75 = tmp73 - tmp74
    tmp77 = 1.0
    tmp78 = tmp76 * tmp77
    tmp79 = 0.0
    tmp80 = triton_helpers.maximum(tmp72, tmp79)
    tmp81 = tmp69 - tmp80
    tmp82 = tmp80 / tmp81
    tmp83 = triton_helpers.maximum(tmp75, tmp79)
    tmp84 = tmp83 - tmp81
    tmp85 = tmp84 / tmp83
    tmp86 = tmp82 - tmp85
    tmp87 = -tmp86
    tmp88 = tmp87 * tmp77
    tmp89 = tmp78 + tmp88
    tl.store(in_out_ptr0 + (x2), tmp89, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4), (16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tgt_spans], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg1_1, buf0, 64, grid=grid(64), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [cost_span], Original ATen: [aten._cdist_forward]
        buf1 = torch.ops.aten._cdist_forward.default(reinterpret_tensor(arg0_1, (16, 4), (4, 1), 0), buf0, 1.0, None)
        buf2 = buf1
        del buf1
        buf6 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [mul_4, areas2, add_2, right, left, sub_4, inter, union, iou, right_1, left_1, sub_6, enclosing_area, sub_7, truediv_1, sub_8, cost_giou, mul_5, C], Original ATen: [aten.mul, aten.sub, aten.add, aten.minimum, aten.maximum, aten.clamp, aten.div, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_maximum_minimum_mul_neg_sub_1.run(buf6, arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del buf0
    buf7 = empty_strided_cpu((4, 4, 16), (64, 16, 1), torch.float32)
    buf7.copy_(reinterpret_tensor(buf6, (4, 4, 16), (64, 16, 1), 0), False)
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
