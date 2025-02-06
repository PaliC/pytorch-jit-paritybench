# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/np/cnpfyuf3pesgmooo2jeq3c6gcwsmxbwwu6zctmdrnsc5lsibbetg.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.elu]
# Source node to ATen node mapping:
#   x_1 => expm1, gt, mul, mul_2, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_2, 0), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul, %mul_2), kwargs = {})
triton_poi_fused_elu_0 = async_compile.triton('triton_poi_fused_elu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yl/cyl5gnut7wx5vkfqaniiknvdljelvxd36s4yecipl6old5csrtky.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_5 => add, add_1, mul_6, mul_7, mul_8, reciprocal, sqrt, sub
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_7, 1e-05), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %primals_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %mul_6), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_8), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %primals_9), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp8 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tmp9 = tmp7 - tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = tmp15 * tmp3
    tmp17 = tmp9 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ho/choetyyi7kqdd2alxwyddlanay2xvsxkztsf3wvmevcnpjv2hv2g.py
# Topologically Sorted Source Nodes: [edges], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   edges => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%bmm_1, %bmm], 2), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (4*x1 + ((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/va/cvaahlzj33h55lrztrr7msd2ybxq6bnw7harci46iawdh7hrnov5.py
# Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   x_13 => div
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%bmm_2, 4), kwargs = {})
triton_poi_fused_div_3 = async_compile.triton('triton_poi_fused_div_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_3(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tx/ctxq4hzrwoq6ottaty6xcymw2pdz5q26laprjrl6fvp764vkiap7.py
# Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_20 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_1, %view_18], 2), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 4, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 8, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (4*x1 + ((-4) + (x0))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp9, tmp11, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 12, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr2 + (4*x1 + ((-8) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp4, tmp19, tmp23)
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4), (4, 1))
    assert_size_stride(primals_11, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_12, (4, 8), (8, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4), (4, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, 4), (4, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 4), (4, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, 12), (12, 1))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, 4), (4, 1))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, 4), (4, 1))
    assert_size_stride(primals_37, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_3, reinterpret_tensor(primals_1, (16, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf0)
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf0, buf1, 64, grid=grid(64), stream=stream0)
        buf2 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf1, (16, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf2)
        del primals_5
        buf3 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_1.run(buf2, primals_6, primals_7, primals_8, primals_9, buf3, 64, grid=grid(64), stream=stream0)
        del primals_9
        buf4 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [receivers], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_10, (4, 4, 4), (0, 4, 1), 0), reinterpret_tensor(buf3, (4, 4, 4), (16, 4, 1), 0), out=buf4)
        buf5 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [senders], Original ATen: [aten.bmm]
        extern_kernels.bmm(primals_11, reinterpret_tensor(buf3, (4, 4, 4), (16, 4, 1), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 4, 8), (32, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [edges], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf5, buf4, buf6, 128, grid=grid(128), stream=stream0)
        buf7 = reinterpret_tensor(buf5, (16, 4), (4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf6, (16, 8), (8, 1), 0), reinterpret_tensor(primals_12, (8, 4), (1, 8), 0), alpha=1, beta=1, out=buf7)
        del primals_13
        buf8 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf7, buf8, 64, grid=grid(64), stream=stream0)
        buf9 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_15, reinterpret_tensor(buf8, (16, 4), (4, 1), 0), reinterpret_tensor(primals_14, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf9)
        del primals_15
        buf10 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_1.run(buf9, primals_16, primals_17, primals_18, primals_19, buf10, 64, grid=grid(64), stream=stream0)
        del primals_19
        buf11 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [incoming], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_10, (4, 4, 4), (0, 1, 4), 0), reinterpret_tensor(buf10, (4, 4, 4), (16, 4, 1), 0), out=buf11)
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_3.run(buf12, 64, grid=grid(64), stream=stream0)
        buf13 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, reinterpret_tensor(buf12, (16, 4), (4, 1), 0), reinterpret_tensor(primals_20, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf13)
        del primals_21
        buf14 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf13, buf14, 64, grid=grid(64), stream=stream0)
        buf15 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_23, reinterpret_tensor(buf14, (16, 4), (4, 1), 0), reinterpret_tensor(primals_22, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf15)
        del primals_23
        buf16 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_1.run(buf15, primals_24, primals_25, primals_26, primals_27, buf16, 64, grid=grid(64), stream=stream0)
        del primals_27
        buf17 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [receivers_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_10, (4, 4, 4), (0, 4, 1), 0), reinterpret_tensor(buf16, (4, 4, 4), (16, 4, 1), 0), out=buf17)
        buf18 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [senders_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(primals_11, reinterpret_tensor(buf16, (4, 4, 4), (16, 4, 1), 0), out=buf18)
        buf19 = empty_strided_cuda((4, 4, 12), (48, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf18, buf17, buf10, buf19, 192, grid=grid(192), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (16, 4), (4, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, reinterpret_tensor(buf19, (16, 12), (12, 1), 0), reinterpret_tensor(primals_28, (12, 4), (1, 12), 0), alpha=1, beta=1, out=buf20)
        del primals_29
        buf21 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf20, buf21, 64, grid=grid(64), stream=stream0)
        buf22 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_31, reinterpret_tensor(buf21, (16, 4), (4, 1), 0), reinterpret_tensor(primals_30, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf22)
        del primals_31
        buf23 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_1.run(buf22, primals_32, primals_33, primals_34, primals_35, buf23, 64, grid=grid(64), stream=stream0)
        del primals_35
        buf24 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, buf23, reinterpret_tensor(primals_36, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf24)
        del primals_37
    return (reinterpret_tensor(buf24, (4, 4, 4), (16, 4, 1), 0), primals_6, primals_7, primals_8, primals_10, primals_16, primals_17, primals_18, primals_24, primals_25, primals_26, primals_32, primals_33, primals_34, reinterpret_tensor(primals_1, (16, 4), (4, 1), 0), buf0, reinterpret_tensor(buf1, (16, 4), (4, 1), 0), buf2, reinterpret_tensor(buf6, (16, 8), (8, 1), 0), buf7, reinterpret_tensor(buf8, (16, 4), (4, 1), 0), buf9, reinterpret_tensor(buf12, (16, 4), (4, 1), 0), buf13, reinterpret_tensor(buf14, (16, 4), (4, 1), 0), buf15, reinterpret_tensor(buf19, (16, 12), (12, 1), 0), buf20, reinterpret_tensor(buf21, (16, 4), (4, 1), 0), buf22, buf23, primals_36, primals_30, primals_28, reinterpret_tensor(primals_11, (4, 4, 4), (16, 1, 4), 0), primals_22, primals_20, primals_14, primals_12, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
