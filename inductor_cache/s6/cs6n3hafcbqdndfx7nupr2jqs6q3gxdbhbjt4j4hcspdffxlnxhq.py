# AOT ID: ['28_forward']
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


# kernel path: inductor_cache/ro/crov34jnt4pylsy3yz5zs2aagst56sinhhyn4u263xv6zbppo2hu.py
# Topologically Sorted Source Nodes: [p], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
# Source node to ATen node mapping:
#   p => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add, add_4, add_5, add_6, clamp_max_2, clamp_max_3, clamp_min, clamp_min_2, clamp_min_3, convert_element_type, convert_element_type_1, convert_element_type_3, iota, mul, mul_2, mul_3, mul_4, sub, sub_2, sub_3, sub_4, sub_5, sub_6
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.int64), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_4), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_4), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 + tmp2
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16 - tmp2
    tmp18 = triton_helpers.maximum(tmp17, tmp6)
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tmp19 + tmp9
    tmp21 = triton_helpers.minimum(tmp20, tmp11)
    tmp22 = tl.load(in_ptr0 + (tmp21 + 4*tmp12 + 16*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (tmp19 + 4*tmp12 + 16*x2), xmask, eviction_policy='evict_last')
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19.to(tl.float32)
    tmp26 = tmp18 - tmp25
    tmp27 = triton_helpers.maximum(tmp26, tmp6)
    tmp28 = 1.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp24 * tmp29
    tmp31 = tmp23 + tmp30
    tmp32 = tl.load(in_ptr0 + (tmp19 + 4*tmp8 + 16*x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (tmp21 + 4*tmp8 + 16*x2), xmask, eviction_policy='evict_last')
    tmp34 = tmp33 - tmp32
    tmp35 = tmp34 * tmp29
    tmp36 = tmp32 + tmp35
    tmp37 = tmp31 - tmp36
    tmp38 = tmp8.to(tl.float32)
    tmp39 = tmp7 - tmp38
    tmp40 = triton_helpers.maximum(tmp39, tmp6)
    tmp41 = triton_helpers.minimum(tmp40, tmp28)
    tmp42 = tmp37 * tmp41
    tmp43 = tmp36 + tmp42
    tl.store(in_out_ptr0 + (x4), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7u/c7u57kyonrqxstobhpygnrvq2lrq5tcgjwwvrqnhb7dokllvinmv.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_8, mul_6, mul_7, sub_7
#   input_3 => gt, mul_8, where
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_6, %primals_2, %primals_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_5), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_8, 0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %add_8), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_8, %mul_8), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
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
    tmp22 = tmp21 * tmp17
    tmp23 = tl.where(tmp19, tmp17, tmp22)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [p], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0.run(buf2, primals_1, 1024, grid=grid(1024), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 8, 8), (256, 64, 8, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 4, 8, 8), (256, 64, 8, 1), torch.float32)
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_convolution_1.run(buf4, buf6, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, 1024, grid=grid(1024), stream=stream0)
        del primals_3
    return (buf6, primals_2, primals_4, primals_5, primals_6, primals_7, primals_8, buf2, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
