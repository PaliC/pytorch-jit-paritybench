# AOT ID: ['6_forward']
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


# kernel path: inductor_cache/ur/curnfiuejirh7p4hzkf47ch6t4hc2jeh7ucs35d2or2voota4yjt.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   y => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%_adaptive_avg_pool2d, %permute], 2), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*(x0) + 16*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + 4*(x0) + 16*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (2 + 4*(x0) + 16*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (3 + 4*(x0) + 16*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 8, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (16*x1 + ((-4) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + (4 + 16*x1 + ((-4) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 + tmp19
    tmp22 = tl.load(in_ptr0 + (8 + 16*x1 + ((-4) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 + tmp21
    tmp24 = tl.load(in_ptr0 + (12 + 16*x1 + ((-4) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp23
    tmp26 = 0.25
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp16, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp15, tmp29)
    tl.store(out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7g/c7g3x2wrm43bscqsuz3442bqpkhugl5qeeyez5aqooeylg3r37sr.py
# Topologically Sorted Source Nodes: [y_1, y_2, add, hardtanh, truediv, y_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add => add_2
#   hardtanh => clamp_max, clamp_min
#   truediv => div
#   y_1 => convolution
#   y_2 => add_1, mul_1, mul_2, sub
#   y_3 => mul_3
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max, 6), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %div), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_hardtanh_mul_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_hardtanh_mul_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_hardtanh_mul_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_hardtanh_mul_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 8) % 8)
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
    tmp18 = 3.0
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = 6.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = 0.16666666666666666
    tmp25 = tmp23 * tmp24
    tmp26 = tmp17 * tmp25
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jj/cjjgzvniqz7odbf3lar234k3k33qlpbnpwwt3vanina36rqkuozh.py
# Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/li/clidcscy3jnx6j5o6uukv75nlunbawuusyxodps3yt6t5z4iimpb.py
# Topologically Sorted Source Nodes: [mul_1, y_4], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_4
#   y_4 => mul_5
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %expand_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %expand), kwargs = {})
triton_poi_fused_mul_3 = async_compile.triton('triton_poi_fused_mul_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (8, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (8, ), (1, ))
    assert_size_stride(primals_4, (8, ), (1, ))
    assert_size_stride(primals_5, (8, ), (1, ))
    assert_size_stride(primals_6, (8, ), (1, ))
    assert_size_stride(primals_7, (8, ), (1, ))
    assert_size_stride(primals_8, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 8, 1), (32, 8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_1, buf0, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 8, 8, 1), (64, 8, 1, 1))
        buf2 = buf1; del buf1  # reuse
        buf3 = empty_strided_cuda((4, 8, 8, 1), (64, 8, 1, 256), torch.float32)
        buf4 = reinterpret_tensor(buf3, (4, 8, 8, 1), (64, 8, 1, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [y_1, y_2, add, hardtanh, truediv, y_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_div_hardtanh_mul_1.run(buf2, buf4, primals_3, primals_4, primals_5, primals_6, primals_7, 256, grid=grid(256), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(reinterpret_tensor(buf4, (4, 8, 4, 1), (64, 8, 1, 0), 0), primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 4, 4, 1), (16, 4, 1, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf6, primals_9, 64, grid=grid(64), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf4, (4, 8, 1, 4), (64, 8, 0, 1), 4), primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 4, 1, 4), (16, 4, 4, 1))
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf8, primals_11, 64, grid=grid(64), stream=stream0)
        del primals_11
        buf9 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, y_4], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_3.run(primals_1, buf8, buf6, buf9, 256, grid=grid(256), stream=stream0)
    return (buf9, primals_1, primals_2, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, buf0, buf2, reinterpret_tensor(buf4, (4, 8, 4, 1), (64, 8, 1, 1), 0), reinterpret_tensor(buf4, (4, 8, 1, 4), (64, 8, 1, 1), 4), buf6, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
