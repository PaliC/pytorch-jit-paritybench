# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/z4/cz42irpnm5aw7r3lk3lbtqaqsmarpa5vdjijducdb3hmcj7dpscg.py
# Topologically Sorted Source Nodes: [pad], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   pad => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_1, %slice_2), kwargs = {})
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %copy, 2, 1, 5), kwargs = {})
#   %slice_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %slice_7, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_2 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %slice_12, 2, 5, 6), kwargs = {})
triton_poi_fused_copy_0 = async_compile.triton('triton_poi_fused_copy_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 6)
    x2 = xindex
    y1 = yindex // 6
    tmp0 = y0
    tmp1 = tl.full([1, 1], 5, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to((-4) + y0, [XBLOCK, YBLOCK])
    tmp4 = tl.full([1, 1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp8 = tl.full([1, 1], 1, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1, 1], 5, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr0 + ((-4) + x2 + 4*y0 + 16*y1), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = float("nan")
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tmp3 >= tmp4
    tmp20 = tl.full([1, 1], 5, tl.int64)
    tmp21 = tmp3 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp2
    tmp24 = tl.load(in_ptr0 + ((-20) + x2 + 4*y0 + 16*y1), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = float("nan")
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tl.where(tmp5, tmp18, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp2, tmp27, tmp28)
    tmp30 = tl.full([1, 1], 1, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.broadcast_to(4 + y0, [XBLOCK, YBLOCK])
    tmp33 = tl.full([1, 1], 1, tl.int64)
    tmp34 = tmp32 >= tmp33
    tmp35 = tl.full([1, 1], 5, tl.int64)
    tmp36 = tmp32 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tmp37 & tmp31
    tmp39 = tl.load(in_ptr0 + (12 + x2 + 4*y0 + 16*y1), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp40 = float("nan")
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp31, tmp41, tmp42)
    tmp44 = tmp0 >= tmp30
    tmp45 = tmp0 < tmp1
    tmp46 = tmp44 & tmp45
    tmp47 = tl.load(in_ptr0 + ((-4) + x2 + 4*y0 + 16*y1), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = float("nan")
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tl.where(tmp31, tmp43, tmp49)
    tmp51 = tl.where(tmp2, tmp29, tmp50)
    tl.store(out_ptr0 + (y0 + 6*x2 + 24*y1), tmp51, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/2x/c2xpqfnepzqhkl4z6rszddt6obq4ye63bphgu5u7rkfaztayaton.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => expm1, gt, mul_3, mul_5, where
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%slice_scatter_default_2, %primals_2, %primals_3, [1], [0], [1], False, [0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_3), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_3,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul_3, %mul_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = tmp17 * tmp11
    tmp21 = libdevice.expm1(tmp20)
    tmp22 = tmp21 * tmp11
    tmp23 = tl.where(tmp19, tmp20, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2h/c2hwwy6f2tac5qwnvmffevi3cknutwf3p5oiyslcfk33hgtndgxa.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_3 => _low_memory_max_pool2d_with_offsets, getitem_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%unsqueeze_4, [1, 3], [1, 2], [0, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_2 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = (-1) + 2*x0
    tmp6 = tmp5 >= tmp0
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp4 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-1) + 2*x2), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp0
    tmp14 = tmp12 < tmp7
    tmp15 = tmp13 & tmp14
    tmp16 = tmp4 & tmp15
    tmp17 = tl.load(in_ptr0 + (2*x2), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = tmp17 > tmp11
    tmp19 = tl.full([1], 1, tl.int8)
    tmp20 = tl.full([1], 0, tl.int8)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp17, tmp11)
    tmp23 = 1 + 2*x0
    tmp24 = tmp23 >= tmp0
    tmp25 = tmp23 < tmp7
    tmp26 = tmp24 & tmp25
    tmp27 = tmp4 & tmp26
    tmp28 = tl.load(in_ptr0 + (1 + 2*x2), tmp27 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp29 = tmp28 > tmp22
    tmp30 = tl.full([1], 2, tl.int8)
    tmp31 = tl.where(tmp29, tmp30, tmp21)
    tmp32 = triton_helpers.maximum(tmp28, tmp22)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
    tl.store(out_ptr1 + (x2), tmp32, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 4, 6), (24, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_0.run(primals_1, buf1, 24, 4, grid=grid(24, 4), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 4), (16, 4, 1))
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_elu_1.run(buf3, buf5, primals_3, primals_4, primals_5, primals_6, primals_7, 64, grid=grid(64), stream=stream0)
        del primals_3
        buf6 = empty_strided_cuda((4, 4, 1, 2), (8, 2, 2, 1), torch.int8)
        buf7 = empty_strided_cuda((4, 4, 1, 2), (8, 2, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_2.run(buf5, buf6, buf7, 32, grid=grid(32), stream=stream0)
    return (reinterpret_tensor(buf7, (4, 2, 4), (8, 1, 2), 0), primals_2, primals_4, primals_5, primals_6, primals_7, buf1, buf3, reinterpret_tensor(buf5, (4, 4, 1, 4), (16, 4, 4, 1), 0), buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
