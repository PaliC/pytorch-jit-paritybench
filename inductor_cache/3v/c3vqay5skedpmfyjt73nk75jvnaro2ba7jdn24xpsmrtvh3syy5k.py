# AOT ID: ['9_inference']
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


# kernel path: inductor_cache/wp/cwpkyu3mud5j5g4s5jxcmq5n2wt6qnbeztdoozpnzpwpogwfysll.py
# Topologically Sorted Source Nodes: [corners, mul, setitem, mul_1, setitem_1, mul_2, setitem_2, mul_3, setitem_3], Original ATen: [aten.zeros, aten.mul, aten.copy]
# Source node to ATen node mapping:
#   corners => full_default
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   setitem => copy
#   setitem_1 => copy_1
#   setitem_2 => copy_2
#   setitem_3 => copy_3
# Graph fragment:
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([4, 2, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, -0.5), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_5, %mul), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %copy, 1, 0), kwargs = {})
#   %select_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %select_scatter_default, 1, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, -0.5), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_12, %mul_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_1, %copy_1, 1, 0), kwargs = {})
#   %select_scatter_default_3 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %select_scatter_default_2, 1, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, -0.5), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_19, %mul_2), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %copy_2, 1, 1), kwargs = {})
#   %select_scatter_default_5 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_3, %select_scatter_default_4, 1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, 0.5), kwargs = {})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_26, %mul_3), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_3, %copy_3, 1, 1), kwargs = {})
#   %select_scatter_default_7 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_5, %select_scatter_default_6, 1, 1), kwargs = {})
triton_poi_fused_copy_mul_zeros_0 = async_compile.triton('triton_poi_fused_copy_mul_zeros_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_mul_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_mul_zeros_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 2)
    x0 = (xindex % 4)
    x2 = xindex // 8
    x4 = xindex
    tmp5 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tmp3 == tmp1
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tmp1 == tmp8
    tmp11 = -0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp8 == tmp1
    tmp14 = tmp3 == tmp8
    tmp15 = tmp5 * tmp11
    tmp16 = 0.0
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.where(tmp9, tmp17, tmp16)
    tmp19 = tl.where(tmp14, tmp15, tmp18)
    tmp20 = tmp8 == tmp8
    tmp21 = tl.where(tmp20, tmp17, tmp16)
    tmp22 = tl.where(tmp13, tmp19, tmp21)
    tmp23 = tl.where(tmp4, tmp12, tmp22)
    tmp24 = tmp1 == tmp1
    tmp25 = tl.where(tmp24, tmp19, tmp18)
    tmp26 = tl.where(tmp9, tmp23, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp28 = tmp0 == tmp8
    tmp29 = tl.where(tmp28, tmp17, tmp16)
    tmp30 = tl.where(tmp2, tmp19, tmp29)
    tmp31 = tl.where(tmp28, tmp23, tmp30)
    tmp32 = tl.where(tmp2, tmp27, tmp31)
    tl.store(out_ptr0 + (x4), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ie/ciesevxwyrejovu2mrtyp3x2m4kodg2jvij4p644gmudq42pfvfc.py
# Topologically Sorted Source Nodes: [mul_4, setitem_4, mul_5, setitem_5], Original ATen: [aten.mul, aten.copy]
# Source node to ATen node mapping:
#   mul_4 => mul_4
#   mul_5 => mul_5
#   setitem_4 => copy_4
#   setitem_5 => copy_5
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, 0.5), kwargs = {})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_33, %mul_4), kwargs = {})
#   %select_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_4, %copy_4, 1, 2), kwargs = {})
#   %select_scatter_default_9 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_7, %select_scatter_default_8, 1, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, 0.5), kwargs = {})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_40, %mul_5), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_5, %copy_5, 1, 2), kwargs = {})
#   %select_scatter_default_11 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_9, %select_scatter_default_10, 1, 1), kwargs = {})
triton_poi_fused_copy_mul_1 = async_compile.triton('triton_poi_fused_copy_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 2)
    x0 = (xindex % 4)
    x2 = xindex // 8
    x4 = xindex
    tmp6 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (x0 + 8*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (4 + x0 + 8*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 2, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp1 == tmp9
    tmp12 = tmp11 * tmp7
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp16 = tl.where(tmp10, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp8, tmp16)
    tmp18 = tmp0 == tmp9
    tmp20 = tl.where(tmp18, tmp14, tmp19)
    tmp21 = tl.where(tmp2, tmp17, tmp20)
    tl.store(out_ptr0 + (x4), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5suu3vu6lkiegp72ti6zqkkn4l2oow5fiftqaidzw3igfrnymn4.py
# Topologically Sorted Source Nodes: [mul_6, setitem_6, mul_7, setitem_7, cat, add], Original ATen: [aten.mul, aten.copy, aten.cat, aten.add]
# Source node to ATen node mapping:
#   add => add
#   cat => cat
#   mul_6 => mul_6
#   mul_7 => mul_7
#   setitem_6 => copy_6
#   setitem_7 => copy_7
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, 0.5), kwargs = {})
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_47, %mul_6), kwargs = {})
#   %select_scatter_default_12 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_6, %copy_6, 1, 3), kwargs = {})
#   %select_scatter_default_13 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_11, %select_scatter_default_12, 1, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, -0.5), kwargs = {})
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_54, %mul_7), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_7, %copy_7, 1, 3), kwargs = {})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_13, %select_scatter_default_14, 1, 1), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_1, %unsqueeze_3], 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_15, %cat), kwargs = {})
triton_poi_fused_add_cat_copy_mul_2 = async_compile.triton('triton_poi_fused_add_cat_copy_mul_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_copy_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_copy_mul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 2)
    x0 = (xindex % 4)
    x2 = xindex // 8
    x4 = xindex
    tmp6 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (x0 + 8*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (4 + x0 + 8*x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 3, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp7 = -0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp1 == tmp9
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp15 = tl.where(tmp5, tmp13, tmp14)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp5, tmp8, tmp17)
    tmp19 = tmp0 == tmp9
    tmp21 = tl.where(tmp19, tmp15, tmp20)
    tmp22 = tl.where(tmp2, tmp18, tmp21)
    tmp23 = tl.full([1], 0, tl.int64)
    tmp24 = tmp0 >= tmp23
    tmp25 = tl.full([1], 1, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr0 + (4*x2), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp0 >= tmp25
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr0 + (1 + 4*x2), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.where(tmp26, tmp27, tmp31)
    tmp33 = tmp22 + tmp32
    tl.store(out_ptr0 + (x4), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [corners, mul, setitem, mul_1, setitem_1, mul_2, setitem_2, mul_3, setitem_3], Original ATen: [aten.zeros, aten.mul, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_mul_zeros_0.run(arg0_1, buf0, 32, grid=grid(32), stream=stream0)
        buf1 = empty_strided_cuda((4, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_4, setitem_4, mul_5, setitem_5], Original ATen: [aten.mul, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_mul_1.run(arg0_1, buf0, buf1, 32, grid=grid(32), stream=stream0)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_6, setitem_6, mul_7, setitem_7, cat, add], Original ATen: [aten.mul, aten.copy, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_copy_mul_2.run(arg0_1, buf1, buf2, 32, grid=grid(32), stream=stream0)
        del arg0_1
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
