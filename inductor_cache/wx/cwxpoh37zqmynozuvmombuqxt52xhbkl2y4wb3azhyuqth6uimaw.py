# AOT ID: ['30_forward']
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


# kernel path: inductor_cache/ky/ckyexssipebz2uwiqywtmfiy2ifjlv4gaxlw552eoqvhzpmv6a7u.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   out => gt, mul, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %mul), kwargs = {})
#   %gt_49 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_0 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/u6/cu6ohmsxaiogsxlmsx34k7uek2dgh2ixvdle7trdkjppdkbqjg7w.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_1 => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 96)
    x0 = (xindex % 4096)
    x2 = xindex // 393216
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.1
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/as/caslfuw3wdzhtxmfstclkncl2f454rp35atbahry4ophxuza3vfk.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_3 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat, %where_1], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 128)
    x0 = (xindex % 4096)
    x2 = xindex // 524288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 393216*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-96) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.1
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/wf/cwf4mjsljomsezgfacv44ldo7dmuamafjknksd2zcjxk6fjxgroh.py
# Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_5 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_1, %where_2], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 160)
    x0 = (xindex % 4096)
    x2 = xindex // 655360
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 524288*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-128) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.1
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/62/c623bbtgmvlyjxmh5esztfgmkxhavcgygaxe5be2qhs4jiqkxidc.py
# Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_7 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %where_3], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 192)
    x0 = (xindex % 4096)
    x2 = xindex // 786432
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 160, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 655360*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-160) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.1
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5spa5yvfeu2jqug242elgvrvkp3lhzepkxzyfnkn5wwwmcefwzt.py
# Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_9 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_3, %where_4], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3670016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 224)
    x0 = (xindex % 4096)
    x2 = xindex // 917504
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 786432*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 224, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-192) + x1) + 131072*x2), tmp6, other=0.0)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.1
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp5, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/pq/cpq2pqwn3ynpzpxjgnhw2h4rjqnu65q75exmsujydkn6v4cml233.py
# Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_11 => add
# Graph fragment:
#   %add : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_6, %convolution), kwargs = {})
triton_poi_fused_add_6 = async_compile.triton('triton_poi_fused_add_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/3e/c3e4oxktlelbb62cze2nntth7xro7se3cjpffh5wmpzaklxeffuc.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   output => cat_25
# Graph fragment:
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add, %add_1, %add_2, %add_3, %add_4, %convolution], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 384)
    x0 = (xindex % 4096)
    x2 = xindex // 1572864
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 262144*x2), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4096*((-128) + x1) + 262144*x2), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 256, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4096*((-192) + x1) + 262144*x2), tmp19, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 320, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4096*((-256) + x1) + 262144*x2), tmp24, other=0.0)
    tmp26 = tl.load(in_ptr3 + (x0 + 4096*((-256) + x1) + 262144*x2), tmp24, other=0.0)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp24, tmp27, tmp28)
    tmp30 = tmp0 >= tmp22
    tmp31 = tl.full([1], 384, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr5 + (x0 + 4096*((-320) + x1) + 262144*x2), tmp30, other=0.0)
    tmp34 = tl.where(tmp24, tmp29, tmp33)
    tmp35 = tl.where(tmp19, tmp20, tmp34)
    tmp36 = tl.where(tmp14, tmp15, tmp35)
    tmp37 = tl.where(tmp9, tmp10, tmp36)
    tmp38 = tl.where(tmp4, tmp5, tmp37)
    tl.store(out_ptr0 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/jy/cjy7xcqrcmn53gy2p5mfg6tcbmumrlzax3uuik2lt4wfst7yytys.py
# Topologically Sorted Source Nodes: [output_3], Original ATen: [aten.pixel_shuffle]
# Source node to ATen node mapping:
#   output_3 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_pixel_shuffle_8 = async_compile.triton('triton_poi_fused_pixel_shuffle_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pixel_shuffle_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pixel_shuffle_8(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 4)
    y2 = ((yindex // 256) % 64)
    y6 = yindex // 16384
    y3 = ((yindex // 16384) % 4)
    y7 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*y2 + 4096*x5 + 16384*y1 + 65536*y6), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + 4*y1 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x5 + 4*y7), tmp2, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35 = args
    args.clear()
    assert_size_stride(primals_1, (64, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_2, (4, 144, 64, 64), (589824, 4096, 64, 1))
    assert_size_stride(primals_3, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_5, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_6, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_7, (32, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_8, (64, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_9, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_11, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_12, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_13, (32, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_14, (64, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_15, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_16, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_17, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_18, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_19, (32, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_20, (64, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_21, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_22, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_23, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_24, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_25, (32, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_26, (64, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_27, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_28, (32, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_29, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_30, (32, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_31, (32, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_32, (64, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_33, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_34, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 64, 64), (262144, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf88 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf1, buf88, 524288, grid=grid(524288), stream=stream0)
        buf2 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf0, buf1, buf2, 1572864, grid=grid(1572864), stream=stream0)
        del buf1
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf87 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf3, buf87, 524288, grid=grid(524288), stream=stream0)
        buf4 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf2, buf3, buf4, 2097152, grid=grid(2097152), stream=stream0)
        del buf3
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf86 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf5, buf86, 524288, grid=grid(524288), stream=stream0)
        buf6 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf4, buf5, buf6, 2621440, grid=grid(2621440), stream=stream0)
        del buf5
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf85 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf7, buf85, 524288, grid=grid(524288), stream=stream0)
        buf8 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf6, buf7, buf8, 3145728, grid=grid(3145728), stream=stream0)
        del buf7
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf84 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf9, buf84, 524288, grid=grid(524288), stream=stream0)
        buf10 = empty_strided_cuda((4, 224, 64, 64), (917504, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf8, buf9, buf10, 3670016, grid=grid(3670016), stream=stream0)
        del buf9
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf12, buf0, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf83 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf13, buf83, 524288, grid=grid(524288), stream=stream0)
        buf14 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf12, buf13, buf14, 1572864, grid=grid(1572864), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf82 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf15, buf82, 524288, grid=grid(524288), stream=stream0)
        buf16 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf14, buf15, buf16, 2097152, grid=grid(2097152), stream=stream0)
        del buf15
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf81 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf17, buf81, 524288, grid=grid(524288), stream=stream0)
        buf18 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf16, buf17, buf18, 2621440, grid=grid(2621440), stream=stream0)
        del buf17
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf80 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf19, buf80, 524288, grid=grid(524288), stream=stream0)
        buf20 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf18, buf19, buf20, 3145728, grid=grid(3145728), stream=stream0)
        del buf19
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf79 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf21, buf79, 524288, grid=grid(524288), stream=stream0)
        buf22 = empty_strided_cuda((4, 224, 64, 64), (917504, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf20, buf21, buf22, 3670016, grid=grid(3670016), stream=stream0)
        del buf21
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf24, buf12, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf78 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf25, buf78, 524288, grid=grid(524288), stream=stream0)
        buf26 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_25], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf24, buf25, buf26, 1572864, grid=grid(1572864), stream=stream0)
        del buf25
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf77 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf27, buf77, 524288, grid=grid(524288), stream=stream0)
        buf28 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf26, buf27, buf28, 2097152, grid=grid(2097152), stream=stream0)
        del buf27
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf76 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf29, buf76, 524288, grid=grid(524288), stream=stream0)
        buf30 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_29], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf28, buf29, buf30, 2621440, grid=grid(2621440), stream=stream0)
        del buf29
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf75 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf31, buf75, 524288, grid=grid(524288), stream=stream0)
        buf32 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf30, buf31, buf32, 3145728, grid=grid(3145728), stream=stream0)
        del buf31
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf74 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_32], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf33, buf74, 524288, grid=grid(524288), stream=stream0)
        buf34 = empty_strided_cuda((4, 224, 64, 64), (917504, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf32, buf33, buf34, 3670016, grid=grid(3670016), stream=stream0)
        del buf33
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf36, buf24, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf73 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf37, buf73, 524288, grid=grid(524288), stream=stream0)
        buf38 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf36, buf37, buf38, 1572864, grid=grid(1572864), stream=stream0)
        del buf37
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf72 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf39, buf72, 524288, grid=grid(524288), stream=stream0)
        buf40 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf38, buf39, buf40, 2097152, grid=grid(2097152), stream=stream0)
        del buf39
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf71 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf41, buf71, 524288, grid=grid(524288), stream=stream0)
        buf42 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf40, buf41, buf42, 2621440, grid=grid(2621440), stream=stream0)
        del buf41
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf70 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf43, buf70, 524288, grid=grid(524288), stream=stream0)
        buf44 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf42, buf43, buf44, 3145728, grid=grid(3145728), stream=stream0)
        del buf43
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf69 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_44], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf45, buf69, 524288, grid=grid(524288), stream=stream0)
        buf46 = empty_strided_cuda((4, 224, 64, 64), (917504, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf44, buf45, buf46, 3670016, grid=grid(3670016), stream=stream0)
        del buf45
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf48, buf36, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf68 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_48], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf49, buf68, 524288, grid=grid(524288), stream=stream0)
        buf50 = empty_strided_cuda((4, 96, 64, 64), (393216, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf48, buf49, buf50, 1572864, grid=grid(1572864), stream=stream0)
        del buf49
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf67 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf51, buf67, 524288, grid=grid(524288), stream=stream0)
        buf52 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_51], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf50, buf51, buf52, 2097152, grid=grid(2097152), stream=stream0)
        del buf51
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf66 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_52], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf53, buf66, 524288, grid=grid(524288), stream=stream0)
        buf54 = empty_strided_cuda((4, 160, 64, 64), (655360, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_53], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf52, buf53, buf54, 2621440, grid=grid(2621440), stream=stream0)
        del buf53
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf65 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf55, buf65, 524288, grid=grid(524288), stream=stream0)
        buf56 = empty_strided_cuda((4, 192, 64, 64), (786432, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_55], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf54, buf55, buf56, 3145728, grid=grid(3145728), stream=stream0)
        del buf55
        # Topologically Sorted Source Nodes: [conv2d_29], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf64 = empty_strided_cuda((4, 32, 64, 64), (131072, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_56], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_0.run(buf57, buf64, 524288, grid=grid(524288), stream=stream0)
        buf58 = empty_strided_cuda((4, 224, 64, 64), (917504, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf56, buf57, buf58, 3670016, grid=grid(3670016), stream=stream0)
        del buf57
        # Topologically Sorted Source Nodes: [out_58], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf60 = empty_strided_cuda((4, 384, 64, 64), (1572864, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf12, buf24, buf36, buf48, buf59, buf0, buf60, 6291456, grid=grid(6291456), stream=stream0)
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 64, 64, 64), (262144, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf63 = reinterpret_tensor(buf59, (4, 4, 64, 4, 64, 4), (262144, 65536, 1024, 256, 4, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [output_3], Original ATen: [aten.pixel_shuffle]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pixel_shuffle_8.run(buf62, primals_35, buf63, 262144, 4, grid=grid(262144, 4), stream=stream0)
        del buf62
        del primals_35
    return (reinterpret_tensor(buf63, (4, 4, 256, 256), (262144, 65536, 256, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, buf0, buf2, buf4, buf6, buf8, buf10, buf12, buf14, buf16, buf18, buf20, buf22, buf24, buf26, buf28, buf30, buf32, buf34, buf36, buf38, buf40, buf42, buf44, buf46, buf48, buf50, buf52, buf54, buf56, buf58, buf60, buf61, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 144, 64, 64), (589824, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
