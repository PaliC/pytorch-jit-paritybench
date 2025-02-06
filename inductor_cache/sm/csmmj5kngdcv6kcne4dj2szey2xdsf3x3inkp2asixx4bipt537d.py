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


# kernel path: inductor_cache/6b/c6by7oej4rqwlld3cacy7ipqohaz6y6lau3wxfvepbxsvb255t2q.py
# Topologically Sorted Source Nodes: [branch5x5], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   branch5x5 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/je/cje6wdzjfwvjk43gktujeuvuzqwqjhdg7vi2gjszfgn56ozx56t4.py
# Topologically Sorted Source Nodes: [branch3x3dbl_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   branch3x3dbl_1 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_3, %primals_10, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 24)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/om/com7hhj3d53kqldcj2vmdzbtbpvmklsrpjoofi2ugabiinmjmkdz.py
# Topologically Sorted Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%primals_3, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_2 = async_compile.triton('triton_poi_fused_avg_pool2d_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + x4), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + x4), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + x4), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + x4), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + x4), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + x4), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x0) + ((-1)*x1) + x0*x1 + ((5) * ((5) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (5)))*((5) * ((5) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (5))) + ((-1)*x0*((5) * ((5) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (5)))) + ((-1)*x1*((5) * ((5) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (5)))) + ((5) * ((5) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (5))) + ((5) * ((5) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (5)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l6/cl64arjnzqqnbnesowta2uh5iqtkmmieabxubeaxen7dv5qf6qel.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_2, %convolution_5, %convolution_6], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 88)
    x0 = (xindex % 16)
    x2 = xindex // 1408
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 256*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 40, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 16*((-16) + x1) + 384*x2), tmp13 & xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-16) + x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 64, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (x0 + 16*((-40) + x1) + 384*x2), tmp22 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr5 + ((-40) + x1), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 88, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr6 + (x0 + 16*((-64) + x1) + 384*x2), tmp28 & xmask, other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-64) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp28, tmp33, tmp34)
    tmp36 = tl.where(tmp22, tmp27, tmp35)
    tmp37 = tl.where(tmp13, tmp18, tmp36)
    tmp38 = tl.where(tmp4, tmp9, tmp37)
    tl.store(out_ptr0 + (x3), tmp38, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (24, 16, 5, 5), (400, 25, 5, 1))
    assert_size_stride(primals_7, (24, ), (1, ))
    assert_size_stride(primals_8, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (24, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (24, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_15, (24, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [branch1x1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 4, 4), (256, 16, 4, 1))
        # Topologically Sorted Source Nodes: [branch5x5], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_3, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 16, 4, 4), (256, 16, 4, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [branch5x5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf2, primals_5, 1024, grid=grid(1024), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [branch5x5_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_6, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 24, 4, 4), (384, 16, 4, 1))
        # Topologically Sorted Source Nodes: [branch3x3dbl], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(primals_3, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 16, 4, 4), (256, 16, 4, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [branch3x3dbl], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf5, primals_9, 1024, grid=grid(1024), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [branch3x3dbl_1], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 24, 4, 4), (384, 16, 4, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [branch3x3dbl_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf7, primals_11, 1536, grid=grid(1536), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [branch3x3dbl_2], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 24, 4, 4), (384, 16, 4, 1))
        buf9 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_2.run(primals_3, buf9, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [branch_pool_1], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 24, 4, 4), (384, 16, 4, 1))
        buf11 = empty_strided_cuda((4, 88, 4, 4), (1408, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf0, primals_2, buf3, primals_7, buf8, primals_13, buf10, primals_15, buf11, 5632, grid=grid(5632), stream=stream0)
        del buf0
        del buf10
        del buf3
        del buf8
        del primals_13
        del primals_15
        del primals_2
        del primals_7
    return (buf11, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, buf2, buf5, buf7, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((24, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((24, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
