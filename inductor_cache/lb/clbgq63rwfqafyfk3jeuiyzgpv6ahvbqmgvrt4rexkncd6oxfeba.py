# AOT ID: ['6_inference']
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


# kernel path: inductor_cache/os/cosyuktozoku3m3aqelpfbvwgiifd673zthm7uckr364fjhg3ypa.py
# Topologically Sorted Source Nodes: [area_p, area_g, add_2, truediv_2, add, truediv_3, add_1, br, truediv, sub, truediv_1, sub_1, tl, sub_2, prod_3, lt, type_1, en, area_i, area_u, add_3, iou, pow_1, loss], Original ATen: [aten.prod, aten.add, aten.div, aten.minimum, aten.sub, aten.maximum, aten.lt, aten._to_copy, aten.mul, aten.pow, aten.rsub]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   area_g => prod_1
#   area_i => mul
#   area_p => prod
#   area_u => sub_3
#   br => minimum
#   en => prod_2
#   iou => div_4
#   loss => sub_4
#   lt => lt
#   pow_1 => pow_1
#   prod_3 => prod_3
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   tl => maximum
#   truediv => div
#   truediv_1 => div_1
#   truediv_2 => div_2
#   truediv_3 => div_3
#   type_1 => convert_element_type
# Graph fragment:
#   %prod : [num_users=1] = call_function[target=torch.ops.aten.prod.dim_int](args = (%slice_18, 1), kwargs = {})
#   %prod_1 : [num_users=1] = call_function[target=torch.ops.aten.prod.dim_int](args = (%slice_20, 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%prod, %prod_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_12, 2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_10, %div_2), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_16, 2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_14, %div_3), kwargs = {})
#   %minimum : [num_users=2] = call_function[target=torch.ops.aten.minimum.default](args = (%add, %add_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_4, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_2, %div), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_8, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_6, %div_1), kwargs = {})
#   %maximum : [num_users=2] = call_function[target=torch.ops.aten.maximum.default](args = (%sub, %sub_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %maximum), kwargs = {})
#   %prod_3 : [num_users=1] = call_function[target=torch.ops.aten.prod.dim_int](args = (%sub_2, 1), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%maximum, %minimum), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %prod_2 : [num_users=1] = call_function[target=torch.ops.aten.prod.dim_int](args = (%convert_element_type, 1), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%prod_3, %prod_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %mul), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_3, 1e-16), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, %add_3), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_4, 2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %pow_1), kwargs = {})
triton_poi_fused__to_copy_add_div_lt_maximum_minimum_mul_pow_prod_rsub_sub_0 = async_compile.triton('triton_poi_fused__to_copy_add_div_lt_maximum_minimum_mul_pow_prod_rsub_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_lt_maximum_minimum_mul_pow_prod_rsub_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_lt_maximum_minimum_mul_pow_prod_rsub_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp6 * tmp2
    tmp8 = tmp5 + tmp7
    tmp9 = triton_helpers.minimum(tmp4, tmp8)
    tmp10 = tmp0 - tmp3
    tmp11 = tmp5 - tmp7
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tmp9 - tmp12
    tmp16 = tmp15 * tmp2
    tmp17 = tmp14 + tmp16
    tmp20 = tmp19 * tmp2
    tmp21 = tmp18 + tmp20
    tmp22 = triton_helpers.minimum(tmp17, tmp21)
    tmp23 = tmp14 - tmp16
    tmp24 = tmp18 - tmp20
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = tmp22 - tmp25
    tmp27 = tmp13 * tmp26
    tmp28 = tmp12 < tmp9
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp25 < tmp22
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp27 * tmp32
    tmp34 = tmp1 * tmp15
    tmp35 = tmp6 * tmp19
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 - tmp33
    tmp38 = 1e-16
    tmp39 = tmp37 + tmp38
    tmp40 = tmp33 / tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = 1.0
    tmp43 = tmp42 - tmp41
    tl.store(in_out_ptr0 + (x0), tmp43, xmask)
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
        buf0 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [area_p, area_g, add_2, truediv_2, add, truediv_3, add_1, br, truediv, sub, truediv_1, sub_1, tl, sub_2, prod_3, lt, type_1, en, area_i, area_u, add_3, iou, pow_1, loss], Original ATen: [aten.prod, aten.add, aten.div, aten.minimum, aten.sub, aten.maximum, aten.lt, aten._to_copy, aten.mul, aten.pow, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_lt_maximum_minimum_mul_pow_prod_rsub_sub_0.run(buf1, arg0_1, arg1_1, 64, grid=grid(64), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf1, )


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
