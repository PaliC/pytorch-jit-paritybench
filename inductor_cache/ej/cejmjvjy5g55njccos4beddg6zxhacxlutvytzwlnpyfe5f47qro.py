# AOT ID: ['3_inference']
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


# kernel path: inductor_cache/uc/cuciabvyr3vzg5g732mn5llqssgm3es3osjbqh2wivohbdbny7cz.py
# Topologically Sorted Source Nodes: [eq, keep, root_cubes_nms], Original ATen: [aten.eq, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   eq => eq
#   keep => convert_element_type
#   root_cubes_nms => mul
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%arg0_1, %getitem), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %arg0_1), kwargs = {})
triton_poi_fused__to_copy_eq_mul_0 = async_compile.triton('triton_poi_fused__to_copy_eq_mul_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_eq_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_eq_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp0
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l5/cl54fluo2d5pxphwedjem3oz4cdxe2pra3rfeyx5r3i3doqghlsh.py
# Topologically Sorted Source Nodes: [human_centers, indices, float_2, sub, truediv, mul_1, add, truediv_1, real_locations, setitem, setitem_1], Original ATen: [aten.zeros, aten.cat, aten._to_copy, aten.sub, aten.div, aten.mul, aten.add, aten.copy]
# Source node to ATen node mapping:
#   add => add
#   float_2 => convert_element_type_1
#   human_centers => full
#   indices => cat
#   mul_1 => mul_1
#   real_locations => sub_1
#   setitem => copy
#   setitem_1 => copy_1
#   sub => sub
#   truediv => div_2
#   truediv_1 => div_3
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([4, 10, 5], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_1, %view_2, %view_3], 2), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, 1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_1, %sub), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %arg2_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg3_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg2_1, 2.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %div_3), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_3, %sub_1), kwargs = {})
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %copy, 2, 0, 3), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_1, %getitem_2), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default, %copy_1, 2, 4), kwargs = {})
triton_poi_fused__to_copy_add_cat_copy_div_mul_sub_zeros_1 = async_compile.triton('triton_poi_fused__to_copy_add_cat_copy_div_mul_sub_zeros_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_copy_div_mul_sub_zeros_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_copy_div_mul_sub_zeros_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = xindex // 5
    x2 = xindex
    tmp54 = tl.load(in_ptr1 + (0))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp59 = tl.load(in_ptr2 + (0))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp63 = tl.load(in_ptr3 + (0))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp76 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr0 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([1], 16, tl.int64)
    tmp11 = tl.where((tmp9 < 0) != (tmp10 < 0), tl.where(tmp9 % tmp10 != 0, tmp9 // tmp10 - 1, tmp9 // tmp10), tmp9 // tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp3 >= tmp6
    tmp15 = tl.full([1], 2, tl.int64)
    tmp16 = tmp3 < tmp15
    tmp17 = tmp14 & tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr0 + (x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full([1], 16, tl.int64)
    tmp21 = tmp19 % tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = tmp21 != tmp22
    tmp24 = (libdevice.signbit(tmp21) != 0) if (tmp21).dtype is tl.float32 else tmp21 < 0
    tmp25 = (libdevice.signbit(tmp20) != 0) if (tmp20).dtype is tl.float32 else tmp20 < 0
    tmp26 = tmp24 != tmp25
    tmp27 = tmp23 & tmp26
    tmp28 = tmp21 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp21)
    tmp30 = tl.full([1], 4, tl.int64)
    tmp31 = tl.where((tmp29 < 0) != (tmp30 < 0), tl.where(tmp29 % tmp30 != 0, tmp29 // tmp30 - 1, tmp29 // tmp30), tmp29 // tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp18, tmp31, tmp32)
    tmp34 = tmp3 >= tmp15
    tmp35 = tl.full([1], 3, tl.int64)
    tmp36 = tmp3 < tmp35
    tmp37 = tmp34 & tmp2
    tmp38 = tl.load(in_ptr0 + (x1), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full([1], 4, tl.int64)
    tmp40 = tmp38 % tmp39
    tmp41 = tl.full([1], 0, tl.int32)
    tmp42 = tmp40 != tmp41
    tmp43 = (libdevice.signbit(tmp40) != 0) if (tmp40).dtype is tl.float32 else tmp40 < 0
    tmp44 = (libdevice.signbit(tmp39) != 0) if (tmp39).dtype is tl.float32 else tmp39 < 0
    tmp45 = tmp43 != tmp44
    tmp46 = tmp42 & tmp45
    tmp47 = tmp40 + tmp39
    tmp48 = tl.where(tmp46, tmp47, tmp40)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp37, tmp48, tmp49)
    tmp51 = tl.where(tmp17, tmp33, tmp50)
    tmp52 = tl.where(tmp7, tmp13, tmp51)
    tmp53 = tmp52.to(tl.float32)
    tmp56 = tmp55 - tmp6
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp53 / tmp57
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp58 * tmp61
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp62 + tmp65
    tmp67 = 0.5
    tmp68 = tmp61 * tmp67
    tmp69 = tmp66 - tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp2, tmp69, tmp70)
    tmp72 = 0.0
    tmp73 = tl.where(tmp2, tmp71, tmp72)
    tmp74 = tl.full([1], 4, tl.int32)
    tmp75 = tmp0 == tmp74
    tmp77 = tl.where(tmp75, tmp76, tmp73)
    tl.store(in_out_ptr0 + (x2), tmp77, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (), ())
    assert_size_stride(arg2_1, (), ())
    assert_size_stride(arg3_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max_pool3d_with_indices]
        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg0_1, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        buf1 = buf0[0]
        del buf0
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [eq, keep, root_cubes_nms], Original ATen: [aten.eq, aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_eq_mul_0.run(buf3, arg0_1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [topk], Original ATen: [aten.topk]
        buf4 = torch.ops.aten.topk.default(reinterpret_tensor(buf3, (4, 64), (64, 1), 0), 10)
        del buf3
        buf5 = buf4[0]
        buf6 = buf4[1]
        del buf4
        buf7 = empty_strided_cuda((4, 10, 5), (50, 5, 1), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [human_centers, indices, float_2, sub, truediv, mul_1, add, truediv_1, real_locations, setitem, setitem_1], Original ATen: [aten.zeros, aten.cat, aten._to_copy, aten.sub, aten.div, aten.mul, aten.add, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_copy_div_mul_sub_zeros_1.run(buf8, buf6, arg1_1, arg2_1, arg3_1, buf5, 200, grid=grid(200), stream=stream0)
        del arg1_1
        del arg2_1
        del arg3_1
        del buf5
        del buf6
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
