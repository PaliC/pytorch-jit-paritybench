# AOT ID: ['17_forward']
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


# kernel path: inductor_cache/yz/cyzu7qcssznrexfrbnssy7sedldleg5piq3qwpg4ekqxzh7oy54k.py
# Topologically Sorted Source Nodes: [x, batch_norm, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub
#   x => convolution
#   x_1 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 8)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/7n/c7nmvac2744x26rxph26jfcitua4jp6bxcqeggd52yqx66k65lbc.py
# Topologically Sorted Source Nodes: [x_14, batch_norm_7, x_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   batch_norm_7 => add_15, mul_22, mul_23, sub_7
#   x_14 => convolution_7
#   x_15 => relu_7
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %primals_44, %primals_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = tmp3 - tmp5
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 * tmp15
    tmp19 = tmp16 * tmp18
    tmp22 = tmp19 + tmp21
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = 0.0
    tmp26 = tmp24 <= tmp25
    tl.store(in_out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr0 + (x0), tmp24, None)
    tl.store(out_ptr1 + (x0), tmp26, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49 = args
    args.clear()
    assert_size_stride(primals_1, (8, 3, 5, 5), (75, 25, 5, 1))
    assert_size_stride(primals_2, (8, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (8, ), (1, ))
    assert_size_stride(primals_5, (8, ), (1, ))
    assert_size_stride(primals_6, (8, ), (1, ))
    assert_size_stride(primals_7, (8, ), (1, ))
    assert_size_stride(primals_8, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, ), (1, ))
    assert_size_stride(primals_12, (8, ), (1, ))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_15, (8, ), (1, ))
    assert_size_stride(primals_16, (8, ), (1, ))
    assert_size_stride(primals_17, (8, ), (1, ))
    assert_size_stride(primals_18, (8, ), (1, ))
    assert_size_stride(primals_19, (8, ), (1, ))
    assert_size_stride(primals_20, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_21, (8, ), (1, ))
    assert_size_stride(primals_22, (8, ), (1, ))
    assert_size_stride(primals_23, (8, ), (1, ))
    assert_size_stride(primals_24, (8, ), (1, ))
    assert_size_stride(primals_25, (8, ), (1, ))
    assert_size_stride(primals_26, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_27, (8, ), (1, ))
    assert_size_stride(primals_28, (8, ), (1, ))
    assert_size_stride(primals_29, (8, ), (1, ))
    assert_size_stride(primals_30, (8, ), (1, ))
    assert_size_stride(primals_31, (8, ), (1, ))
    assert_size_stride(primals_32, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_33, (8, ), (1, ))
    assert_size_stride(primals_34, (8, ), (1, ))
    assert_size_stride(primals_35, (8, ), (1, ))
    assert_size_stride(primals_36, (8, ), (1, ))
    assert_size_stride(primals_37, (8, ), (1, ))
    assert_size_stride(primals_38, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_39, (8, ), (1, ))
    assert_size_stride(primals_40, (8, ), (1, ))
    assert_size_stride(primals_41, (8, ), (1, ))
    assert_size_stride(primals_42, (8, ), (1, ))
    assert_size_stride(primals_43, (8, ), (1, ))
    assert_size_stride(primals_44, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_45, (1, ), (1, ))
    assert_size_stride(primals_46, (1, ), (1, ))
    assert_size_stride(primals_47, (1, ), (1, ))
    assert_size_stride(primals_48, (1, ), (1, ))
    assert_size_stride(primals_49, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 8, 64, 64), (32768, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 8, 64, 64), (32768, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, batch_norm, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_2, primals_4, primals_5, primals_6, primals_7, buf2, 131072, grid=grid(131072), stream=stream0)
        del primals_2
        del primals_7
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_8, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 8, 64, 64), (32768, 4096, 64, 1))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 8, 64, 64), (32768, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, batch_norm_1, x_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf4, primals_9, primals_10, primals_11, primals_12, primals_13, buf5, 131072, grid=grid(131072), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_14, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 8, 64, 64), (32768, 4096, 64, 1))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((4, 8, 64, 64), (32768, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, batch_norm_2, x_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf7, primals_15, primals_16, primals_17, primals_18, primals_19, buf8, 131072, grid=grid(131072), stream=stream0)
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_20, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 8, 64, 64), (32768, 4096, 64, 1))
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((4, 8, 64, 64), (32768, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, batch_norm_3, x_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf10, primals_21, primals_22, primals_23, primals_24, primals_25, buf11, 131072, grid=grid(131072), stream=stream0)
        del primals_21
        del primals_25
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_26, stride=(1, 1), padding=(8, 8), dilation=(8, 8), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 8, 64, 64), (32768, 4096, 64, 1))
        buf13 = buf12; del buf12  # reuse
        buf14 = empty_strided_cuda((4, 8, 64, 64), (32768, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, batch_norm_4, x_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf13, primals_27, primals_28, primals_29, primals_30, primals_31, buf14, 131072, grid=grid(131072), stream=stream0)
        del primals_27
        del primals_31
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_32, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 8, 64, 64), (32768, 4096, 64, 1))
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided_cuda((4, 8, 64, 64), (32768, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_10, batch_norm_5, x_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf16, primals_33, primals_34, primals_35, primals_36, primals_37, buf17, 131072, grid=grid(131072), stream=stream0)
        del primals_33
        del primals_37
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_38, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 8, 64, 64), (32768, 4096, 64, 1))
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided_cuda((4, 8, 64, 64), (32768, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12, batch_norm_6, x_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf19, primals_39, primals_40, primals_41, primals_42, primals_43, buf20, 131072, grid=grid(131072), stream=stream0)
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 1, 64, 64), (4096, 4096, 64, 1))
        buf22 = buf21; del buf21  # reuse
        buf23 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.float32)
        buf24 = empty_strided_cuda((4, 1, 64, 64), (4096, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_14, batch_norm_7, x_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_threshold_backward_1.run(buf22, primals_45, primals_46, primals_47, primals_48, primals_49, buf23, buf24, 16384, grid=grid(16384), stream=stream0)
        del primals_45
        del primals_49
    return (buf23, primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, buf22, buf24, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 3, 5, 5), (75, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((8, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
