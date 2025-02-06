# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/l2/cl2cp4v7znty3olp2nmmgmreavkmaof4pjob6a6qlrzyzerdfbk4.py
# Topologically Sorted Source Nodes: [y, y_1, y_2], Original ATen: [aten.stack, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.view_as_complex]
# Source node to ATen node mapping:
#   y => cat
#   y_1 => abs_1, gt, mul_4, mul_5, sign, sub_1, where
#   y_2 => view_as_complex
# Graph fragment:
#   %cat : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_13, %unsqueeze_14], -1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%cat,), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.01), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%cat,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.01), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, 0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub_1, %mul_5), kwargs = {})
#   %view_as_complex : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%where,), kwargs = {})
triton_poi_fused_abs_gt_mul_sign_stack_sub_view_as_complex_where_0 = async_compile.triton('triton_poi_fused_abs_gt_mul_sign_stack_sub_view_as_complex_where_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_gt_mul_sign_stack_sub_view_as_complex_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_gt_mul_sign_stack_sub_view_as_complex_where_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x2 = xindex // 8
    x1 = ((xindex // 2) % 4)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (5*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr0 + (1 + 2*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (5*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 2, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr0 + (1 + 2*x2), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr1 + (5*x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr0 + (2*x2), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr2 + (5*x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = tl.load(in_ptr4 + (x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp18, tmp31, tmp32)
    tmp34 = tl.where(tmp4, tmp17, tmp33)
    tmp35 = tl_math.abs(tmp34)
    tmp36 = 0.01
    tmp37 = tmp35 > tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = tmp38 < tmp34
    tmp40 = tmp39.to(tl.int8)
    tmp41 = tmp34 < tmp38
    tmp42 = tmp41.to(tl.int8)
    tmp43 = tmp40 - tmp42
    tmp44 = tmp43.to(tmp34.dtype)
    tmp45 = tmp44 * tmp36
    tmp46 = tmp34 - tmp45
    tmp47 = 0.0
    tmp48 = tmp34 * tmp47
    tmp49 = tl.where(tmp37, tmp46, tmp48)
    tl.store(out_ptr0 + (x3), tmp37, xmask)
    tl.store(in_out_ptr0 + (x3), tmp49, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/su/csudynjco35filpopjergr7ruxlqxh3zmlvbutgkyfzjbffpbw73.py
# Topologically Sorted Source Nodes: [x_4, reshape], Original ATen: [aten.add, aten.clone]
# Source node to ATen node mapping:
#   reshape => clone
#   x_4 => add_3
# Graph fragment:
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_fft_c2r, %unsqueeze), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_1 = async_compile.triton('triton_poi_fused_add_clone_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 4)
    y4 = yindex // 4
    y1 = ((yindex // 4) % 4)
    y2 = yindex // 16
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1 + 4*y0 + 16*y2), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3 + 4*y5), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jp/cjpgcjugo5fozatwnhcygov4rlhobfblfmq5dro2pqkr7odcrnqo.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   input_2 => gt_1, mul_6, where_1
# Graph fragment:
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_2, 0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.01), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %view_2, %mul_6), kwargs = {})
triton_poi_fused_leaky_relu_2 = async_compile.triton('triton_poi_fused_leaky_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 16), (16, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4), (4, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._fft_r2c]
        buf0 = torch.ops.aten._fft_r2c.default(reinterpret_tensor(primals_1, (4, 4, 4, 1), (16, 1, 4, 0), 0), [2], 1, True)
        buf1 = buf0
        del buf0
        # Topologically Sorted Source Nodes: [getattr_1], Original ATen: [aten.view_as_real]
        buf2 = torch.ops.aten.view_as_real.default(buf1)
        buf3 = buf2
        buf4 = empty_strided_cuda((4, 4, 3, 4, 2), (96, 24, 8, 2, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 4, 3, 4, 2), (96, 24, 8, 2, 1), torch.bool)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [y, y_1, y_2], Original ATen: [aten.stack, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.view_as_complex]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_gt_mul_sign_stack_sub_view_as_complex_where_0.run(buf6, buf3, primals_2, primals_3, primals_4, primals_5, buf5, 384, grid=grid(384), stream=stream0)
        # Topologically Sorted Source Nodes: [y_1, y_2], Original ATen: [aten.sign, aten.mul, aten.sub, aten.where, aten.view_as_complex]
        buf7 = torch.ops.aten.view_as_complex.default(buf6)
        buf8 = buf7
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._fft_c2r]
        buf9 = torch.ops.aten._fft_c2r.default(buf8, [2], 1, 4)
        del buf6
        del buf7
        del buf8
        buf10 = buf9
        del buf9
        buf11 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, reshape], Original ATen: [aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_1.run(buf10, primals_1, buf11, 64, 4, grid=grid(64, 4), stream=stream0)
        del buf10
        del primals_1
        buf12 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (16, 16), (16, 1), 0), reinterpret_tensor(primals_6, (16, 4), (1, 16), 0), out=buf12)
        buf13 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        buf14 = reinterpret_tensor(buf12, (4, 4, 4), (16, 4, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_2.run(buf14, primals_7, buf13, 64, grid=grid(64), stream=stream0)
        del primals_7
        buf15 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf14, (16, 4), (4, 1), 0), reinterpret_tensor(primals_8, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf15)
        del primals_9
    return (reinterpret_tensor(buf15, (4, 4, 4), (16, 1, 4), 0), primals_4, primals_5, buf3, reinterpret_tensor(primals_2, (4, ), (5, ), 0), reinterpret_tensor(primals_3, (4, ), (5, ), 0), buf5, reinterpret_tensor(buf11, (16, 16), (16, 1), 0), buf13, reinterpret_tensor(buf14, (16, 4), (4, 1), 0), primals_8, primals_6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
