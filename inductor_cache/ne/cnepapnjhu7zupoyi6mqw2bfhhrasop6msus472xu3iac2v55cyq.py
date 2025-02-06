# AOT ID: ['73_forward']
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


# kernel path: inductor_cache/a5/ca5jtjpxj2rtri2ldrf22z4mcme3r76xwbz2kzlesrtyxdtyrx37.py
# Topologically Sorted Source Nodes: [out1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out1 => convolution
# Graph fragment:
#   %convolution : [num_users=6] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/bq/cbqv67zeifjedvytjr27cgogdaivzazm7cqyipomsala4epf4eby.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_1 => convolution_1
#   input_2 => gt
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
triton_poi_fused_convolution_leaky_relu_1 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/fi/cfix37ggvgcbmfkh5n7qrn5st2rlohowwnl4g5prm4mlhq7nh4ez.py
# Topologically Sorted Source Nodes: [inputs], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   inputs => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %where], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 128)
    x0 = (xindex % 16)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 1024*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-64) + x1) + 1024*x2), tmp6, other=0.0).to(tl.int1)
    tmp10 = tl.load(in_ptr2 + (x0 + 16*((-64) + x1) + 1024*x2), tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + ((-64) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.01
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp9, tmp12, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/sk/csknmca7gwgbtvpa6zaclsfdmh7yuegrdgyxxktcore5c5krrlau.py
# Topologically Sorted Source Nodes: [inputs_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   inputs_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat, %where_1], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 192)
    x0 = (xindex % 16)
    x2 = xindex // 3072
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 2048*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-128) + x1) + 1024*x2), tmp6, other=0.0).to(tl.int1)
    tmp10 = tl.load(in_ptr2 + (x0 + 16*((-128) + x1) + 1024*x2), tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + ((-128) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.01
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp9, tmp12, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/4d/c4ds4sfgoxuxc3ug2vqpcjrmx3jk54e6iti2fzwurjdywmyf34q7.py
# Topologically Sorted Source Nodes: [inputs_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   inputs_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_1, %where_2], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 256)
    x0 = (xindex % 16)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 3072*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-192) + x1) + 1024*x2), tmp6, other=0.0).to(tl.int1)
    tmp10 = tl.load(in_ptr2 + (x0 + 16*((-192) + x1) + 1024*x2), tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + ((-192) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.01
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp9, tmp12, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqsvvjfub4xoedodwit3gkvyyms3rnv7u7f475nor3uyitp6d26w.py
# Topologically Sorted Source Nodes: [inputs_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   inputs_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %where_3], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 320)
    x0 = (xindex % 16)
    x2 = xindex // 5120
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 4096*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 320, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 16*((-256) + x1) + 1024*x2), tmp6, other=0.0).to(tl.int1)
    tmp10 = tl.load(in_ptr2 + (x0 + 16*((-256) + x1) + 1024*x2), tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + ((-256) + x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.01
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp9, tmp12, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/rs/crsueb674vvg56rgws7oorkeyfebjc2kywzqslgkim4c6zaotbcl.py
# Topologically Sorted Source Nodes: [input_9, mul, input_10], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_10 => add
#   input_9 => convolution_5
#   mul => mul_4
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_12, %primals_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_5, 0.2), kwargs = {})
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_6 = async_compile.triton('triton_poi_fused_add_convolution_mul_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/o3/co3mxlubb6yclrifqhccpkfaptv4njoebblzaj6ducwv5zr7g7cr.py
# Topologically Sorted Source Nodes: [input_29, mul_2, input_30, mul_3, input_31], Original ATen: [aten.convolution, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_29 => convolution_15
#   input_30 => add_2
#   input_31 => add_3
#   mul_2 => mul_14
#   mul_3 => mul_15
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_13, %primals_32, %primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 0.2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %add_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.2), kwargs = {})
#   %add_3 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %convolution), kwargs = {})
triton_poi_fused_add_convolution_mul_7 = async_compile.triton('triton_poi_fused_add_convolution_mul_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp3
    tmp9 = tmp7 + tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/pl/cpls4jutphydbfm3zqdi7fnujtxh5p5nfor3m7npeovmunfocbv2.py
# Topologically Sorted Source Nodes: [out2, out], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   out => add_64
#   out2 => convolution_241
# Graph fragment:
#   %convolution_241 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_63, %primals_484, %primals_485, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_64 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_241), kwargs = {})
triton_poi_fused_add_convolution_8 = async_compile.triton('triton_poi_fused_add_convolution_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/gs/cgsps77nz3pu676ewfs77duz5y5cmdrpprxw2slizgpboezgm3v6.py
# Topologically Sorted Source Nodes: [input_497, input_498, input_499], Original ATen: [aten.convolution, aten.leaky_relu, aten.pixel_shuffle]
# Source node to ATen node mapping:
#   input_497 => convolution_242
#   input_498 => gt_192
#   input_499 => clone
# Graph fragment:
#   %convolution_242 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_64, %primals_486, %primals_487, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_192 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_242, 0), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_convolution_leaky_relu_pixel_shuffle_9 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_pixel_shuffle_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_pixel_shuffle_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_pixel_shuffle_9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y8 = yindex
    y0 = (yindex % 256)
    x6 = (xindex % 4)
    x7 = xindex // 4
    y3 = (yindex % 2)
    y4 = ((yindex // 2) % 2)
    y9 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y8), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2 + 16*y8), tmp4, xmask)
    tl.store(out_ptr1 + (y3 + 2*x6 + 8*y4 + 16*x7 + 64*y9), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jq/cjq7fqozuqqoevvjxwtrriuldvdd2aplp322qhxzxhlleuxrvre3.py
# Topologically Sorted Source Nodes: [input_500, input_501, input_502], Original ATen: [aten.convolution, aten.leaky_relu, aten.pixel_shuffle]
# Source node to ATen node mapping:
#   input_500 => convolution_243
#   input_501 => gt_193
#   input_502 => clone_2
# Graph fragment:
#   %convolution_243 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_1, %primals_488, %primals_489, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_193 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_243, 0), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_convolution_leaky_relu_pixel_shuffle_10 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_pixel_shuffle_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_pixel_shuffle_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_pixel_shuffle_10(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y8 = yindex
    y0 = (yindex % 256)
    x6 = (xindex % 8)
    x7 = xindex // 8
    y3 = (yindex % 2)
    y4 = ((yindex // 2) % 2)
    y9 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y8), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2 + 64*y8), tmp4, xmask)
    tl.store(out_ptr1 + (y3 + 2*x6 + 16*y4 + 32*x7 + 256*y9), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mc/cmclwmvxinvtc6pcanmzkaw44bupu2vzbajlihqb6bw3xcvacqvx.py
# Topologically Sorted Source Nodes: [input_503, input_504], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_503 => convolution_244
#   input_504 => gt_194, mul_258, where_194
# Graph fragment:
#   %convolution_244 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_3, %primals_490, %primals_491, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_194 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_244, 0), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_244, 0.01), kwargs = {})
#   %where_194 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_194, %convolution_244, %mul_258), kwargs = {})
triton_poi_fused_convolution_leaky_relu_11 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_11(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/sd/csde2aebg273i7j37vky4khk3pexppwngfjj7fqt6pjtwoqd6ps2.py
# Topologically Sorted Source Nodes: [input_505], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_505 => convolution_245
# Graph fragment:
#   %convolution_245 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_194, %primals_492, %primals_493, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493 = args
    args.clear()
    assert_size_stride(primals_1, (64, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_87, (64, ), (1, ))
    assert_size_stride(primals_88, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_97, (64, ), (1, ))
    assert_size_stride(primals_98, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_105, (64, ), (1, ))
    assert_size_stride(primals_106, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_107, (64, ), (1, ))
    assert_size_stride(primals_108, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_109, (64, ), (1, ))
    assert_size_stride(primals_110, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_111, (64, ), (1, ))
    assert_size_stride(primals_112, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_117, (64, ), (1, ))
    assert_size_stride(primals_118, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_119, (64, ), (1, ))
    assert_size_stride(primals_120, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_125, (64, ), (1, ))
    assert_size_stride(primals_126, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_128, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_131, (64, ), (1, ))
    assert_size_stride(primals_132, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_135, (64, ), (1, ))
    assert_size_stride(primals_136, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_137, (64, ), (1, ))
    assert_size_stride(primals_138, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_141, (64, ), (1, ))
    assert_size_stride(primals_142, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_147, (64, ), (1, ))
    assert_size_stride(primals_148, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_149, (64, ), (1, ))
    assert_size_stride(primals_150, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_157, (64, ), (1, ))
    assert_size_stride(primals_158, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_159, (64, ), (1, ))
    assert_size_stride(primals_160, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_161, (64, ), (1, ))
    assert_size_stride(primals_162, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_165, (64, ), (1, ))
    assert_size_stride(primals_166, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_167, (64, ), (1, ))
    assert_size_stride(primals_168, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_175, (64, ), (1, ))
    assert_size_stride(primals_176, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_177, (64, ), (1, ))
    assert_size_stride(primals_178, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_179, (64, ), (1, ))
    assert_size_stride(primals_180, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_183, (64, ), (1, ))
    assert_size_stride(primals_184, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_191, (64, ), (1, ))
    assert_size_stride(primals_192, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_193, (64, ), (1, ))
    assert_size_stride(primals_194, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_195, (64, ), (1, ))
    assert_size_stride(primals_196, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_197, (64, ), (1, ))
    assert_size_stride(primals_198, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_199, (64, ), (1, ))
    assert_size_stride(primals_200, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_201, (64, ), (1, ))
    assert_size_stride(primals_202, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_205, (64, ), (1, ))
    assert_size_stride(primals_206, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_207, (64, ), (1, ))
    assert_size_stride(primals_208, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_211, (64, ), (1, ))
    assert_size_stride(primals_212, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_213, (64, ), (1, ))
    assert_size_stride(primals_214, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_215, (64, ), (1, ))
    assert_size_stride(primals_216, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_217, (64, ), (1, ))
    assert_size_stride(primals_218, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_219, (64, ), (1, ))
    assert_size_stride(primals_220, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_225, (64, ), (1, ))
    assert_size_stride(primals_226, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_227, (64, ), (1, ))
    assert_size_stride(primals_228, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_231, (64, ), (1, ))
    assert_size_stride(primals_232, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_233, (64, ), (1, ))
    assert_size_stride(primals_234, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_237, (64, ), (1, ))
    assert_size_stride(primals_238, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_241, (64, ), (1, ))
    assert_size_stride(primals_242, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_243, (64, ), (1, ))
    assert_size_stride(primals_244, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_247, (64, ), (1, ))
    assert_size_stride(primals_248, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_249, (64, ), (1, ))
    assert_size_stride(primals_250, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_251, (64, ), (1, ))
    assert_size_stride(primals_252, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_257, (64, ), (1, ))
    assert_size_stride(primals_258, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_261, (64, ), (1, ))
    assert_size_stride(primals_262, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_263, (64, ), (1, ))
    assert_size_stride(primals_264, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_265, (64, ), (1, ))
    assert_size_stride(primals_266, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_267, (64, ), (1, ))
    assert_size_stride(primals_268, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_269, (64, ), (1, ))
    assert_size_stride(primals_270, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_275, (64, ), (1, ))
    assert_size_stride(primals_276, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_277, (64, ), (1, ))
    assert_size_stride(primals_278, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_279, (64, ), (1, ))
    assert_size_stride(primals_280, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_281, (64, ), (1, ))
    assert_size_stride(primals_282, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_283, (64, ), (1, ))
    assert_size_stride(primals_284, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_285, (64, ), (1, ))
    assert_size_stride(primals_286, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_287, (64, ), (1, ))
    assert_size_stride(primals_288, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_289, (64, ), (1, ))
    assert_size_stride(primals_290, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_291, (64, ), (1, ))
    assert_size_stride(primals_292, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_293, (64, ), (1, ))
    assert_size_stride(primals_294, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_295, (64, ), (1, ))
    assert_size_stride(primals_296, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_297, (64, ), (1, ))
    assert_size_stride(primals_298, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_299, (64, ), (1, ))
    assert_size_stride(primals_300, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_301, (64, ), (1, ))
    assert_size_stride(primals_302, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_303, (64, ), (1, ))
    assert_size_stride(primals_304, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_305, (64, ), (1, ))
    assert_size_stride(primals_306, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_307, (64, ), (1, ))
    assert_size_stride(primals_308, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_309, (64, ), (1, ))
    assert_size_stride(primals_310, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_311, (64, ), (1, ))
    assert_size_stride(primals_312, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_317, (64, ), (1, ))
    assert_size_stride(primals_318, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_321, (64, ), (1, ))
    assert_size_stride(primals_322, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_323, (64, ), (1, ))
    assert_size_stride(primals_324, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_325, (64, ), (1, ))
    assert_size_stride(primals_326, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_327, (64, ), (1, ))
    assert_size_stride(primals_328, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_329, (64, ), (1, ))
    assert_size_stride(primals_330, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_331, (64, ), (1, ))
    assert_size_stride(primals_332, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_333, (64, ), (1, ))
    assert_size_stride(primals_334, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_335, (64, ), (1, ))
    assert_size_stride(primals_336, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_337, (64, ), (1, ))
    assert_size_stride(primals_338, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_345, (64, ), (1, ))
    assert_size_stride(primals_346, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_347, (64, ), (1, ))
    assert_size_stride(primals_348, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_349, (64, ), (1, ))
    assert_size_stride(primals_350, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_351, (64, ), (1, ))
    assert_size_stride(primals_352, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_353, (64, ), (1, ))
    assert_size_stride(primals_354, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_355, (64, ), (1, ))
    assert_size_stride(primals_356, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_357, (64, ), (1, ))
    assert_size_stride(primals_358, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_359, (64, ), (1, ))
    assert_size_stride(primals_360, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_361, (64, ), (1, ))
    assert_size_stride(primals_362, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_363, (64, ), (1, ))
    assert_size_stride(primals_364, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_365, (64, ), (1, ))
    assert_size_stride(primals_366, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_367, (64, ), (1, ))
    assert_size_stride(primals_368, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_371, (64, ), (1, ))
    assert_size_stride(primals_372, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_373, (64, ), (1, ))
    assert_size_stride(primals_374, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_375, (64, ), (1, ))
    assert_size_stride(primals_376, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_377, (64, ), (1, ))
    assert_size_stride(primals_378, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_379, (64, ), (1, ))
    assert_size_stride(primals_380, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_381, (64, ), (1, ))
    assert_size_stride(primals_382, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_383, (64, ), (1, ))
    assert_size_stride(primals_384, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_385, (64, ), (1, ))
    assert_size_stride(primals_386, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_387, (64, ), (1, ))
    assert_size_stride(primals_388, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_389, (64, ), (1, ))
    assert_size_stride(primals_390, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_391, (64, ), (1, ))
    assert_size_stride(primals_392, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_393, (64, ), (1, ))
    assert_size_stride(primals_394, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_395, (64, ), (1, ))
    assert_size_stride(primals_396, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_397, (64, ), (1, ))
    assert_size_stride(primals_398, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_399, (64, ), (1, ))
    assert_size_stride(primals_400, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_401, (64, ), (1, ))
    assert_size_stride(primals_402, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_403, (64, ), (1, ))
    assert_size_stride(primals_404, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_405, (64, ), (1, ))
    assert_size_stride(primals_406, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_407, (64, ), (1, ))
    assert_size_stride(primals_408, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_409, (64, ), (1, ))
    assert_size_stride(primals_410, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_411, (64, ), (1, ))
    assert_size_stride(primals_412, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_413, (64, ), (1, ))
    assert_size_stride(primals_414, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_415, (64, ), (1, ))
    assert_size_stride(primals_416, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_417, (64, ), (1, ))
    assert_size_stride(primals_418, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_419, (64, ), (1, ))
    assert_size_stride(primals_420, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_421, (64, ), (1, ))
    assert_size_stride(primals_422, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_423, (64, ), (1, ))
    assert_size_stride(primals_424, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_425, (64, ), (1, ))
    assert_size_stride(primals_426, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_427, (64, ), (1, ))
    assert_size_stride(primals_428, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_429, (64, ), (1, ))
    assert_size_stride(primals_430, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_431, (64, ), (1, ))
    assert_size_stride(primals_432, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_433, (64, ), (1, ))
    assert_size_stride(primals_434, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_435, (64, ), (1, ))
    assert_size_stride(primals_436, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_437, (64, ), (1, ))
    assert_size_stride(primals_438, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_439, (64, ), (1, ))
    assert_size_stride(primals_440, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_441, (64, ), (1, ))
    assert_size_stride(primals_442, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_443, (64, ), (1, ))
    assert_size_stride(primals_444, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_445, (64, ), (1, ))
    assert_size_stride(primals_446, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_447, (64, ), (1, ))
    assert_size_stride(primals_448, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_449, (64, ), (1, ))
    assert_size_stride(primals_450, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_451, (64, ), (1, ))
    assert_size_stride(primals_452, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_453, (64, ), (1, ))
    assert_size_stride(primals_454, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_455, (64, ), (1, ))
    assert_size_stride(primals_456, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_457, (64, ), (1, ))
    assert_size_stride(primals_458, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_459, (64, ), (1, ))
    assert_size_stride(primals_460, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_461, (64, ), (1, ))
    assert_size_stride(primals_462, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_463, (64, ), (1, ))
    assert_size_stride(primals_464, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_465, (64, ), (1, ))
    assert_size_stride(primals_466, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_467, (64, ), (1, ))
    assert_size_stride(primals_468, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_469, (64, ), (1, ))
    assert_size_stride(primals_470, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_471, (64, ), (1, ))
    assert_size_stride(primals_472, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_473, (64, ), (1, ))
    assert_size_stride(primals_474, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_475, (64, ), (1, ))
    assert_size_stride(primals_476, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_477, (64, ), (1, ))
    assert_size_stride(primals_478, (64, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_479, (64, ), (1, ))
    assert_size_stride(primals_480, (64, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_481, (64, ), (1, ))
    assert_size_stride(primals_482, (64, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_483, (64, ), (1, ))
    assert_size_stride(primals_484, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_485, (64, ), (1, ))
    assert_size_stride(primals_486, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_487, (256, ), (1, ))
    assert_size_stride(primals_488, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_489, (256, ), (1, ))
    assert_size_stride(primals_490, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_491, (64, ), (1, ))
    assert_size_stride(primals_492, (4, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_493, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf1, primals_2, 4096, grid=grid(4096), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf3 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf2, primals_5, buf3, 4096, grid=grid(4096), stream=stream0)
        buf4 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf1, buf3, buf2, primals_5, buf4, 8192, grid=grid(8192), stream=stream0)
        del buf2
        del primals_5
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf6 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf5, primals_7, buf6, 4096, grid=grid(4096), stream=stream0)
        buf7 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf4, buf6, buf5, primals_7, buf7, 12288, grid=grid(12288), stream=stream0)
        del buf5
        del primals_7
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf9 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf8, primals_9, buf9, 4096, grid=grid(4096), stream=stream0)
        buf10 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf7, buf9, buf8, primals_9, buf10, 16384, grid=grid(16384), stream=stream0)
        del buf8
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf12 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf11, primals_11, buf12, 4096, grid=grid(4096), stream=stream0)
        buf13 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf10, buf12, buf11, primals_11, buf13, 20480, grid=grid(20480), stream=stream0)
        del buf11
        del primals_11
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_9, mul, input_10], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf15, primals_13, buf1, 4096, grid=grid(4096), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf17 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf16, primals_15, buf17, 4096, grid=grid(4096), stream=stream0)
        buf18 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf15, buf17, buf16, primals_15, buf18, 8192, grid=grid(8192), stream=stream0)
        del buf16
        del primals_15
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf20 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf19, primals_17, buf20, 4096, grid=grid(4096), stream=stream0)
        buf21 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf18, buf20, buf19, primals_17, buf21, 12288, grid=grid(12288), stream=stream0)
        del buf19
        del primals_17
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf23 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf22, primals_19, buf23, 4096, grid=grid(4096), stream=stream0)
        buf24 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf21, buf23, buf22, primals_19, buf24, 16384, grid=grid(16384), stream=stream0)
        del buf22
        del primals_19
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf26 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf25, primals_21, buf26, 4096, grid=grid(4096), stream=stream0)
        buf27 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf24, buf26, buf25, primals_21, buf27, 20480, grid=grid(20480), stream=stream0)
        del buf25
        del primals_21
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_19, mul_1, input_20], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf29, primals_23, buf15, 4096, grid=grid(4096), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf31 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf30, primals_25, buf31, 4096, grid=grid(4096), stream=stream0)
        buf32 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf29, buf31, buf30, primals_25, buf32, 8192, grid=grid(8192), stream=stream0)
        del buf30
        del primals_25
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf34 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf33, primals_27, buf34, 4096, grid=grid(4096), stream=stream0)
        buf35 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf32, buf34, buf33, primals_27, buf35, 12288, grid=grid(12288), stream=stream0)
        del buf33
        del primals_27
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf37 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf36, primals_29, buf37, 4096, grid=grid(4096), stream=stream0)
        buf38 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf35, buf37, buf36, primals_29, buf38, 16384, grid=grid(16384), stream=stream0)
        del buf36
        del primals_29
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf40 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf39, primals_31, buf40, 4096, grid=grid(4096), stream=stream0)
        buf41 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf38, buf40, buf39, primals_31, buf41, 20480, grid=grid(20480), stream=stream0)
        del buf39
        del primals_31
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_29, mul_2, input_30, mul_3, input_31], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf43, primals_33, buf29, buf1, 4096, grid=grid(4096), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf45 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf44, primals_35, buf45, 4096, grid=grid(4096), stream=stream0)
        buf46 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf43, buf45, buf44, primals_35, buf46, 8192, grid=grid(8192), stream=stream0)
        del buf44
        del primals_35
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf48 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf47, primals_37, buf48, 4096, grid=grid(4096), stream=stream0)
        buf49 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf46, buf48, buf47, primals_37, buf49, 12288, grid=grid(12288), stream=stream0)
        del buf47
        del primals_37
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf51 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf50, primals_39, buf51, 4096, grid=grid(4096), stream=stream0)
        buf52 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf49, buf51, buf50, primals_39, buf52, 16384, grid=grid(16384), stream=stream0)
        del buf50
        del primals_39
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf54 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf53, primals_41, buf54, 4096, grid=grid(4096), stream=stream0)
        buf55 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf52, buf54, buf53, primals_41, buf55, 20480, grid=grid(20480), stream=stream0)
        del buf53
        del primals_41
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_40, mul_4, input_41], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf57, primals_43, buf43, 4096, grid=grid(4096), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf59 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf58, primals_45, buf59, 4096, grid=grid(4096), stream=stream0)
        buf60 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf57, buf59, buf58, primals_45, buf60, 8192, grid=grid(8192), stream=stream0)
        del buf58
        del primals_45
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf62 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf61, primals_47, buf62, 4096, grid=grid(4096), stream=stream0)
        buf63 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf60, buf62, buf61, primals_47, buf63, 12288, grid=grid(12288), stream=stream0)
        del buf61
        del primals_47
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf65 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_46, input_47], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf64, primals_49, buf65, 4096, grid=grid(4096), stream=stream0)
        buf66 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_22], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf63, buf65, buf64, primals_49, buf66, 16384, grid=grid(16384), stream=stream0)
        del buf64
        del primals_49
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf68 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf67, primals_51, buf68, 4096, grid=grid(4096), stream=stream0)
        buf69 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_23], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf66, buf68, buf67, primals_51, buf69, 20480, grid=grid(20480), stream=stream0)
        del buf67
        del primals_51
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_50, mul_5, input_51], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf71, primals_53, buf57, 4096, grid=grid(4096), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf73 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf72, primals_55, buf73, 4096, grid=grid(4096), stream=stream0)
        buf74 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_25], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf71, buf73, buf72, primals_55, buf74, 8192, grid=grid(8192), stream=stream0)
        del buf72
        del primals_55
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf76 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf75, primals_57, buf76, 4096, grid=grid(4096), stream=stream0)
        buf77 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_26], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf74, buf76, buf75, primals_57, buf77, 12288, grid=grid(12288), stream=stream0)
        del buf75
        del primals_57
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf79 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf78, primals_59, buf79, 4096, grid=grid(4096), stream=stream0)
        buf80 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_27], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf77, buf79, buf78, primals_59, buf80, 16384, grid=grid(16384), stream=stream0)
        del buf78
        del primals_59
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf82 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_58, input_59], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf81, primals_61, buf82, 4096, grid=grid(4096), stream=stream0)
        buf83 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf80, buf82, buf81, primals_61, buf83, 20480, grid=grid(20480), stream=stream0)
        del buf81
        del primals_61
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_60, mul_6, input_61, mul_7, input_62], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf85, primals_63, buf71, buf43, 4096, grid=grid(4096), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf87 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_63, input_64], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf86, primals_65, buf87, 4096, grid=grid(4096), stream=stream0)
        buf88 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf85, buf87, buf86, primals_65, buf88, 8192, grid=grid(8192), stream=stream0)
        del buf86
        del primals_65
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf90 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_65, input_66], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf89, primals_67, buf90, 4096, grid=grid(4096), stream=stream0)
        buf91 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_31], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf88, buf90, buf89, primals_67, buf91, 12288, grid=grid(12288), stream=stream0)
        del buf89
        del primals_67
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf93 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_67, input_68], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf92, primals_69, buf93, 4096, grid=grid(4096), stream=stream0)
        buf94 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_32], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf91, buf93, buf92, primals_69, buf94, 16384, grid=grid(16384), stream=stream0)
        del buf92
        del primals_69
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf96 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_69, input_70], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf95, primals_71, buf96, 4096, grid=grid(4096), stream=stream0)
        buf97 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf94, buf96, buf95, primals_71, buf97, 20480, grid=grid(20480), stream=stream0)
        del buf95
        del primals_71
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [input_71, mul_8, input_72], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf99, primals_73, buf85, 4096, grid=grid(4096), stream=stream0)
        del primals_73
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf101 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf100, primals_75, buf101, 4096, grid=grid(4096), stream=stream0)
        buf102 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_35], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf99, buf101, buf100, primals_75, buf102, 8192, grid=grid(8192), stream=stream0)
        del buf100
        del primals_75
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf104 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_75, input_76], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf103, primals_77, buf104, 4096, grid=grid(4096), stream=stream0)
        buf105 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf102, buf104, buf103, primals_77, buf105, 12288, grid=grid(12288), stream=stream0)
        del buf103
        del primals_77
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf107 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf106, primals_79, buf107, 4096, grid=grid(4096), stream=stream0)
        buf108 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_37], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf105, buf107, buf106, primals_79, buf108, 16384, grid=grid(16384), stream=stream0)
        del buf106
        del primals_79
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf110 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_79, input_80], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf109, primals_81, buf110, 4096, grid=grid(4096), stream=stream0)
        buf111 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_38], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf108, buf110, buf109, primals_81, buf111, 20480, grid=grid(20480), stream=stream0)
        del buf109
        del primals_81
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [input_81, mul_9, input_82], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf113, primals_83, buf99, 4096, grid=grid(4096), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf115 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_83, input_84], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf114, primals_85, buf115, 4096, grid=grid(4096), stream=stream0)
        buf116 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf113, buf115, buf114, primals_85, buf116, 8192, grid=grid(8192), stream=stream0)
        del buf114
        del primals_85
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf118 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_85, input_86], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf117, primals_87, buf118, 4096, grid=grid(4096), stream=stream0)
        buf119 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_41], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf116, buf118, buf117, primals_87, buf119, 12288, grid=grid(12288), stream=stream0)
        del buf117
        del primals_87
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf121 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_87, input_88], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf120, primals_89, buf121, 4096, grid=grid(4096), stream=stream0)
        buf122 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_42], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf119, buf121, buf120, primals_89, buf122, 16384, grid=grid(16384), stream=stream0)
        del buf120
        del primals_89
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf124 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf123, primals_91, buf124, 4096, grid=grid(4096), stream=stream0)
        buf125 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_43], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf122, buf124, buf123, primals_91, buf125, 20480, grid=grid(20480), stream=stream0)
        del buf123
        del primals_91
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_91, mul_10, input_92, mul_11, input_93], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf127, primals_93, buf113, buf85, 4096, grid=grid(4096), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf129 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_94, input_95], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf128, primals_95, buf129, 4096, grid=grid(4096), stream=stream0)
        buf130 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_45], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf127, buf129, buf128, primals_95, buf130, 8192, grid=grid(8192), stream=stream0)
        del buf128
        del primals_95
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf132 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_96, input_97], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf131, primals_97, buf132, 4096, grid=grid(4096), stream=stream0)
        buf133 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_46], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf130, buf132, buf131, primals_97, buf133, 12288, grid=grid(12288), stream=stream0)
        del buf131
        del primals_97
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf135 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_98, input_99], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf134, primals_99, buf135, 4096, grid=grid(4096), stream=stream0)
        buf136 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_47], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf133, buf135, buf134, primals_99, buf136, 16384, grid=grid(16384), stream=stream0)
        del buf134
        del primals_99
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf138 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_100, input_101], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf137, primals_101, buf138, 4096, grid=grid(4096), stream=stream0)
        buf139 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_48], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf136, buf138, buf137, primals_101, buf139, 20480, grid=grid(20480), stream=stream0)
        del buf137
        del primals_101
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [input_102, mul_12, input_103], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf141, primals_103, buf127, 4096, grid=grid(4096), stream=stream0)
        del primals_103
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf143 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf142, primals_105, buf143, 4096, grid=grid(4096), stream=stream0)
        buf144 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_50], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf141, buf143, buf142, primals_105, buf144, 8192, grid=grid(8192), stream=stream0)
        del buf142
        del primals_105
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf146 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_106, input_107], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf145, primals_107, buf146, 4096, grid=grid(4096), stream=stream0)
        buf147 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_51], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf144, buf146, buf145, primals_107, buf147, 12288, grid=grid(12288), stream=stream0)
        del buf145
        del primals_107
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf149 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf148, primals_109, buf149, 4096, grid=grid(4096), stream=stream0)
        buf150 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_52], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf147, buf149, buf148, primals_109, buf150, 16384, grid=grid(16384), stream=stream0)
        del buf148
        del primals_109
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf152 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf151, primals_111, buf152, 4096, grid=grid(4096), stream=stream0)
        buf153 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_53], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf150, buf152, buf151, primals_111, buf153, 20480, grid=grid(20480), stream=stream0)
        del buf151
        del primals_111
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [input_112, mul_13, input_113], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf155, primals_113, buf141, 4096, grid=grid(4096), stream=stream0)
        del primals_113
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf157 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_114, input_115], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf156, primals_115, buf157, 4096, grid=grid(4096), stream=stream0)
        buf158 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_55], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf155, buf157, buf156, primals_115, buf158, 8192, grid=grid(8192), stream=stream0)
        del buf156
        del primals_115
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf160 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_116, input_117], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf159, primals_117, buf160, 4096, grid=grid(4096), stream=stream0)
        buf161 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_56], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf158, buf160, buf159, primals_117, buf161, 12288, grid=grid(12288), stream=stream0)
        del buf159
        del primals_117
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf163 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_118, input_119], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf162, primals_119, buf163, 4096, grid=grid(4096), stream=stream0)
        buf164 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_57], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf161, buf163, buf162, primals_119, buf164, 16384, grid=grid(16384), stream=stream0)
        del buf162
        del primals_119
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf166 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_120, input_121], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf165, primals_121, buf166, 4096, grid=grid(4096), stream=stream0)
        buf167 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_58], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf164, buf166, buf165, primals_121, buf167, 20480, grid=grid(20480), stream=stream0)
        del buf165
        del primals_121
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [input_122, mul_14, input_123, mul_15, input_124], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf169, primals_123, buf155, buf127, 4096, grid=grid(4096), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf171 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_125, input_126], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf170, primals_125, buf171, 4096, grid=grid(4096), stream=stream0)
        buf172 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_60], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf169, buf171, buf170, primals_125, buf172, 8192, grid=grid(8192), stream=stream0)
        del buf170
        del primals_125
        # Topologically Sorted Source Nodes: [input_127], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf174 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_127, input_128], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf173, primals_127, buf174, 4096, grid=grid(4096), stream=stream0)
        buf175 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_61], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf172, buf174, buf173, primals_127, buf175, 12288, grid=grid(12288), stream=stream0)
        del buf173
        del primals_127
        # Topologically Sorted Source Nodes: [input_129], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf177 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_129, input_130], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf176, primals_129, buf177, 4096, grid=grid(4096), stream=stream0)
        buf178 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_62], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf175, buf177, buf176, primals_129, buf178, 16384, grid=grid(16384), stream=stream0)
        del buf176
        del primals_129
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf180 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_131, input_132], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf179, primals_131, buf180, 4096, grid=grid(4096), stream=stream0)
        buf181 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_63], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf178, buf180, buf179, primals_131, buf181, 20480, grid=grid(20480), stream=stream0)
        del buf179
        del primals_131
        # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [input_133, mul_16, input_134], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf183, primals_133, buf169, 4096, grid=grid(4096), stream=stream0)
        del primals_133
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf185 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_135, input_136], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf184, primals_135, buf185, 4096, grid=grid(4096), stream=stream0)
        buf186 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_65], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf183, buf185, buf184, primals_135, buf186, 8192, grid=grid(8192), stream=stream0)
        del buf184
        del primals_135
        # Topologically Sorted Source Nodes: [input_137], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf188 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_137, input_138], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf187, primals_137, buf188, 4096, grid=grid(4096), stream=stream0)
        buf189 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_66], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf186, buf188, buf187, primals_137, buf189, 12288, grid=grid(12288), stream=stream0)
        del buf187
        del primals_137
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf191 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_139, input_140], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf190, primals_139, buf191, 4096, grid=grid(4096), stream=stream0)
        buf192 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_67], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf189, buf191, buf190, primals_139, buf192, 16384, grid=grid(16384), stream=stream0)
        del buf190
        del primals_139
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf194 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_141, input_142], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf193, primals_141, buf194, 4096, grid=grid(4096), stream=stream0)
        buf195 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_68], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf192, buf194, buf193, primals_141, buf195, 20480, grid=grid(20480), stream=stream0)
        del buf193
        del primals_141
        # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_143, mul_17, input_144], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf197, primals_143, buf183, 4096, grid=grid(4096), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf199 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_145, input_146], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf198, primals_145, buf199, 4096, grid=grid(4096), stream=stream0)
        buf200 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_70], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf197, buf199, buf198, primals_145, buf200, 8192, grid=grid(8192), stream=stream0)
        del buf198
        del primals_145
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf202 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_147, input_148], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf201, primals_147, buf202, 4096, grid=grid(4096), stream=stream0)
        buf203 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_71], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf200, buf202, buf201, primals_147, buf203, 12288, grid=grid(12288), stream=stream0)
        del buf201
        del primals_147
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf205 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_149, input_150], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf204, primals_149, buf205, 4096, grid=grid(4096), stream=stream0)
        buf206 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_72], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf203, buf205, buf204, primals_149, buf206, 16384, grid=grid(16384), stream=stream0)
        del buf204
        del primals_149
        # Topologically Sorted Source Nodes: [input_151], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf208 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_151, input_152], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf207, primals_151, buf208, 4096, grid=grid(4096), stream=stream0)
        buf209 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_73], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf206, buf208, buf207, primals_151, buf209, 20480, grid=grid(20480), stream=stream0)
        del buf207
        del primals_151
        # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [input_153, mul_18, input_154, mul_19, input_155], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf211, primals_153, buf197, buf169, 4096, grid=grid(4096), stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf213 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_156, input_157], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf212, primals_155, buf213, 4096, grid=grid(4096), stream=stream0)
        buf214 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_75], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf211, buf213, buf212, primals_155, buf214, 8192, grid=grid(8192), stream=stream0)
        del buf212
        del primals_155
        # Topologically Sorted Source Nodes: [input_158], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf216 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_158, input_159], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf215, primals_157, buf216, 4096, grid=grid(4096), stream=stream0)
        buf217 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_76], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf214, buf216, buf215, primals_157, buf217, 12288, grid=grid(12288), stream=stream0)
        del buf215
        del primals_157
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_158, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf219 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_160, input_161], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf218, primals_159, buf219, 4096, grid=grid(4096), stream=stream0)
        buf220 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_77], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf217, buf219, buf218, primals_159, buf220, 16384, grid=grid(16384), stream=stream0)
        del buf218
        del primals_159
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf222 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_162, input_163], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf221, primals_161, buf222, 4096, grid=grid(4096), stream=stream0)
        buf223 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_78], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf220, buf222, buf221, primals_161, buf223, 20480, grid=grid(20480), stream=stream0)
        del buf221
        del primals_161
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf225 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [input_164, mul_20, input_165], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf225, primals_163, buf211, 4096, grid=grid(4096), stream=stream0)
        del primals_163
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf227 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_166, input_167], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf226, primals_165, buf227, 4096, grid=grid(4096), stream=stream0)
        buf228 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_80], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf225, buf227, buf226, primals_165, buf228, 8192, grid=grid(8192), stream=stream0)
        del buf226
        del primals_165
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_166, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf230 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_168, input_169], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf229, primals_167, buf230, 4096, grid=grid(4096), stream=stream0)
        buf231 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_81], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf228, buf230, buf229, primals_167, buf231, 12288, grid=grid(12288), stream=stream0)
        del buf229
        del primals_167
        # Topologically Sorted Source Nodes: [input_170], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_168, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf233 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_170, input_171], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf232, primals_169, buf233, 4096, grid=grid(4096), stream=stream0)
        buf234 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_82], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf231, buf233, buf232, primals_169, buf234, 16384, grid=grid(16384), stream=stream0)
        del buf232
        del primals_169
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf236 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_172, input_173], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf235, primals_171, buf236, 4096, grid=grid(4096), stream=stream0)
        buf237 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_83], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf234, buf236, buf235, primals_171, buf237, 20480, grid=grid(20480), stream=stream0)
        del buf235
        del primals_171
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [input_174, mul_21, input_175], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf239, primals_173, buf225, 4096, grid=grid(4096), stream=stream0)
        del primals_173
        # Topologically Sorted Source Nodes: [input_176], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf241 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_176, input_177], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf240, primals_175, buf241, 4096, grid=grid(4096), stream=stream0)
        buf242 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_85], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf239, buf241, buf240, primals_175, buf242, 8192, grid=grid(8192), stream=stream0)
        del buf240
        del primals_175
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf244 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_178, input_179], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf243, primals_177, buf244, 4096, grid=grid(4096), stream=stream0)
        buf245 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_86], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf242, buf244, buf243, primals_177, buf245, 12288, grid=grid(12288), stream=stream0)
        del buf243
        del primals_177
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_178, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf247 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_180, input_181], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf246, primals_179, buf247, 4096, grid=grid(4096), stream=stream0)
        buf248 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_87], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf245, buf247, buf246, primals_179, buf248, 16384, grid=grid(16384), stream=stream0)
        del buf246
        del primals_179
        # Topologically Sorted Source Nodes: [input_182], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, primals_180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf250 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_182, input_183], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf249, primals_181, buf250, 4096, grid=grid(4096), stream=stream0)
        buf251 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_88], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf248, buf250, buf249, primals_181, buf251, 20480, grid=grid(20480), stream=stream0)
        del buf249
        del primals_181
        # Topologically Sorted Source Nodes: [input_184], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [input_184, mul_22, input_185, mul_23, input_186], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf253, primals_183, buf239, buf211, 4096, grid=grid(4096), stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [input_187], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_184, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf255 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_187, input_188], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf254, primals_185, buf255, 4096, grid=grid(4096), stream=stream0)
        buf256 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_90], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf253, buf255, buf254, primals_185, buf256, 8192, grid=grid(8192), stream=stream0)
        del buf254
        del primals_185
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf258 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_189, input_190], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf257, primals_187, buf258, 4096, grid=grid(4096), stream=stream0)
        buf259 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_91], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf256, buf258, buf257, primals_187, buf259, 12288, grid=grid(12288), stream=stream0)
        del buf257
        del primals_187
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_188, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf261 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_191, input_192], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf260, primals_189, buf261, 4096, grid=grid(4096), stream=stream0)
        buf262 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_92], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf259, buf261, buf260, primals_189, buf262, 16384, grid=grid(16384), stream=stream0)
        del buf260
        del primals_189
        # Topologically Sorted Source Nodes: [input_193], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf264 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_193, input_194], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf263, primals_191, buf264, 4096, grid=grid(4096), stream=stream0)
        buf265 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_93], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf262, buf264, buf263, primals_191, buf265, 20480, grid=grid(20480), stream=stream0)
        del buf263
        del primals_191
        # Topologically Sorted Source Nodes: [input_195], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf267 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [input_195, mul_24, input_196], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf267, primals_193, buf253, 4096, grid=grid(4096), stream=stream0)
        del primals_193
        # Topologically Sorted Source Nodes: [input_197], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_194, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf269 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_197, input_198], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf268, primals_195, buf269, 4096, grid=grid(4096), stream=stream0)
        buf270 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_95], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf267, buf269, buf268, primals_195, buf270, 8192, grid=grid(8192), stream=stream0)
        del buf268
        del primals_195
        # Topologically Sorted Source Nodes: [input_199], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf272 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_199, input_200], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf271, primals_197, buf272, 4096, grid=grid(4096), stream=stream0)
        buf273 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_96], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf270, buf272, buf271, primals_197, buf273, 12288, grid=grid(12288), stream=stream0)
        del buf271
        del primals_197
        # Topologically Sorted Source Nodes: [input_201], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_198, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf275 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_201, input_202], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf274, primals_199, buf275, 4096, grid=grid(4096), stream=stream0)
        buf276 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_97], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf273, buf275, buf274, primals_199, buf276, 16384, grid=grid(16384), stream=stream0)
        del buf274
        del primals_199
        # Topologically Sorted Source Nodes: [input_203], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf278 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_203, input_204], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf277, primals_201, buf278, 4096, grid=grid(4096), stream=stream0)
        buf279 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_98], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf276, buf278, buf277, primals_201, buf279, 20480, grid=grid(20480), stream=stream0)
        del buf277
        del primals_201
        # Topologically Sorted Source Nodes: [input_205], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf281 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [input_205, mul_25, input_206], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf281, primals_203, buf267, 4096, grid=grid(4096), stream=stream0)
        del primals_203
        # Topologically Sorted Source Nodes: [input_207], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_204, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf283 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_207, input_208], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf282, primals_205, buf283, 4096, grid=grid(4096), stream=stream0)
        buf284 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_100], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf281, buf283, buf282, primals_205, buf284, 8192, grid=grid(8192), stream=stream0)
        del buf282
        del primals_205
        # Topologically Sorted Source Nodes: [input_209], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf286 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_209, input_210], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf285, primals_207, buf286, 4096, grid=grid(4096), stream=stream0)
        buf287 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_101], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf284, buf286, buf285, primals_207, buf287, 12288, grid=grid(12288), stream=stream0)
        del buf285
        del primals_207
        # Topologically Sorted Source Nodes: [input_211], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf289 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_211, input_212], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf288, primals_209, buf289, 4096, grid=grid(4096), stream=stream0)
        buf290 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_102], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf287, buf289, buf288, primals_209, buf290, 16384, grid=grid(16384), stream=stream0)
        del buf288
        del primals_209
        # Topologically Sorted Source Nodes: [input_213], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_210, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf292 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_213, input_214], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf291, primals_211, buf292, 4096, grid=grid(4096), stream=stream0)
        buf293 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_103], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf290, buf292, buf291, primals_211, buf293, 20480, grid=grid(20480), stream=stream0)
        del buf291
        del primals_211
        # Topologically Sorted Source Nodes: [input_215], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf295 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [input_215, mul_26, input_216, mul_27, input_217], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf295, primals_213, buf281, buf253, 4096, grid=grid(4096), stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [input_218], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf297 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_218, input_219], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf296, primals_215, buf297, 4096, grid=grid(4096), stream=stream0)
        buf298 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_105], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf295, buf297, buf296, primals_215, buf298, 8192, grid=grid(8192), stream=stream0)
        del buf296
        del primals_215
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf300 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_220, input_221], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf299, primals_217, buf300, 4096, grid=grid(4096), stream=stream0)
        buf301 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_106], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf298, buf300, buf299, primals_217, buf301, 12288, grid=grid(12288), stream=stream0)
        del buf299
        del primals_217
        # Topologically Sorted Source Nodes: [input_222], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf303 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_222, input_223], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf302, primals_219, buf303, 4096, grid=grid(4096), stream=stream0)
        buf304 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_107], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf301, buf303, buf302, primals_219, buf304, 16384, grid=grid(16384), stream=stream0)
        del buf302
        del primals_219
        # Topologically Sorted Source Nodes: [input_224], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_220, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf306 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_224, input_225], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf305, primals_221, buf306, 4096, grid=grid(4096), stream=stream0)
        buf307 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_108], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf304, buf306, buf305, primals_221, buf307, 20480, grid=grid(20480), stream=stream0)
        del buf305
        del primals_221
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf309 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [input_226, mul_28, input_227], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf309, primals_223, buf295, 4096, grid=grid(4096), stream=stream0)
        del primals_223
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf311 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_228, input_229], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf310, primals_225, buf311, 4096, grid=grid(4096), stream=stream0)
        buf312 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_110], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf309, buf311, buf310, primals_225, buf312, 8192, grid=grid(8192), stream=stream0)
        del buf310
        del primals_225
        # Topologically Sorted Source Nodes: [input_230], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf314 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_230, input_231], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf313, primals_227, buf314, 4096, grid=grid(4096), stream=stream0)
        buf315 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_111], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf312, buf314, buf313, primals_227, buf315, 12288, grid=grid(12288), stream=stream0)
        del buf313
        del primals_227
        # Topologically Sorted Source Nodes: [input_232], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, primals_228, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf317 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_232, input_233], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf316, primals_229, buf317, 4096, grid=grid(4096), stream=stream0)
        buf318 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_112], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf315, buf317, buf316, primals_229, buf318, 16384, grid=grid(16384), stream=stream0)
        del buf316
        del primals_229
        # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf320 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_234, input_235], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf319, primals_231, buf320, 4096, grid=grid(4096), stream=stream0)
        buf321 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_113], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf318, buf320, buf319, primals_231, buf321, 20480, grid=grid(20480), stream=stream0)
        del buf319
        del primals_231
        # Topologically Sorted Source Nodes: [input_236], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf323 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [input_236, mul_29, input_237], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf323, primals_233, buf309, 4096, grid=grid(4096), stream=stream0)
        del primals_233
        # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_234, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf325 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_238, input_239], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf324, primals_235, buf325, 4096, grid=grid(4096), stream=stream0)
        buf326 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_115], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf323, buf325, buf324, primals_235, buf326, 8192, grid=grid(8192), stream=stream0)
        del buf324
        del primals_235
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_236, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf328 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_240, input_241], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf327, primals_237, buf328, 4096, grid=grid(4096), stream=stream0)
        buf329 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_116], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf326, buf328, buf327, primals_237, buf329, 12288, grid=grid(12288), stream=stream0)
        del buf327
        del primals_237
        # Topologically Sorted Source Nodes: [input_242], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf331 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_242, input_243], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf330, primals_239, buf331, 4096, grid=grid(4096), stream=stream0)
        buf332 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_117], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf329, buf331, buf330, primals_239, buf332, 16384, grid=grid(16384), stream=stream0)
        del buf330
        del primals_239
        # Topologically Sorted Source Nodes: [input_244], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_240, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf334 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_244, input_245], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf333, primals_241, buf334, 4096, grid=grid(4096), stream=stream0)
        buf335 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_118], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf332, buf334, buf333, primals_241, buf335, 20480, grid=grid(20480), stream=stream0)
        del buf333
        del primals_241
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf337 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [input_246, mul_30, input_247, mul_31, input_248], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf337, primals_243, buf323, buf295, 4096, grid=grid(4096), stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, primals_244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf339 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_249, input_250], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf338, primals_245, buf339, 4096, grid=grid(4096), stream=stream0)
        buf340 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_120], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf337, buf339, buf338, primals_245, buf340, 8192, grid=grid(8192), stream=stream0)
        del buf338
        del primals_245
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, primals_246, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf342 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_251, input_252], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf341, primals_247, buf342, 4096, grid=grid(4096), stream=stream0)
        buf343 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_121], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf340, buf342, buf341, primals_247, buf343, 12288, grid=grid(12288), stream=stream0)
        del buf341
        del primals_247
        # Topologically Sorted Source Nodes: [input_253], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_248, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf345 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_253, input_254], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf344, primals_249, buf345, 4096, grid=grid(4096), stream=stream0)
        buf346 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_122], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf343, buf345, buf344, primals_249, buf346, 16384, grid=grid(16384), stream=stream0)
        del buf344
        del primals_249
        # Topologically Sorted Source Nodes: [input_255], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf348 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_255, input_256], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf347, primals_251, buf348, 4096, grid=grid(4096), stream=stream0)
        buf349 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_123], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf346, buf348, buf347, primals_251, buf349, 20480, grid=grid(20480), stream=stream0)
        del buf347
        del primals_251
        # Topologically Sorted Source Nodes: [input_257], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf351 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [input_257, mul_32, input_258], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf351, primals_253, buf337, 4096, grid=grid(4096), stream=stream0)
        del primals_253
        # Topologically Sorted Source Nodes: [input_259], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_254, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf353 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_259, input_260], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf352, primals_255, buf353, 4096, grid=grid(4096), stream=stream0)
        buf354 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_125], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf351, buf353, buf352, primals_255, buf354, 8192, grid=grid(8192), stream=stream0)
        del buf352
        del primals_255
        # Topologically Sorted Source Nodes: [input_261], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, primals_256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf356 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_261, input_262], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf355, primals_257, buf356, 4096, grid=grid(4096), stream=stream0)
        buf357 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_126], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf354, buf356, buf355, primals_257, buf357, 12288, grid=grid(12288), stream=stream0)
        del buf355
        del primals_257
        # Topologically Sorted Source Nodes: [input_263], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_258, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf359 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_263, input_264], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf358, primals_259, buf359, 4096, grid=grid(4096), stream=stream0)
        buf360 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_127], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf357, buf359, buf358, primals_259, buf360, 16384, grid=grid(16384), stream=stream0)
        del buf358
        del primals_259
        # Topologically Sorted Source Nodes: [input_265], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_260, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf362 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_265, input_266], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf361, primals_261, buf362, 4096, grid=grid(4096), stream=stream0)
        buf363 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_128], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf360, buf362, buf361, primals_261, buf363, 20480, grid=grid(20480), stream=stream0)
        del buf361
        del primals_261
        # Topologically Sorted Source Nodes: [input_267], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf365 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [input_267, mul_33, input_268], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf365, primals_263, buf351, 4096, grid=grid(4096), stream=stream0)
        del primals_263
        # Topologically Sorted Source Nodes: [input_269], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_264, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf367 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_269, input_270], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf366, primals_265, buf367, 4096, grid=grid(4096), stream=stream0)
        buf368 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_130], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf365, buf367, buf366, primals_265, buf368, 8192, grid=grid(8192), stream=stream0)
        del buf366
        del primals_265
        # Topologically Sorted Source Nodes: [input_271], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, primals_266, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf370 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_271, input_272], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf369, primals_267, buf370, 4096, grid=grid(4096), stream=stream0)
        buf371 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_131], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf368, buf370, buf369, primals_267, buf371, 12288, grid=grid(12288), stream=stream0)
        del buf369
        del primals_267
        # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf373 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_273, input_274], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf372, primals_269, buf373, 4096, grid=grid(4096), stream=stream0)
        buf374 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_132], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf371, buf373, buf372, primals_269, buf374, 16384, grid=grid(16384), stream=stream0)
        del buf372
        del primals_269
        # Topologically Sorted Source Nodes: [input_275], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_270, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf376 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_275, input_276], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf375, primals_271, buf376, 4096, grid=grid(4096), stream=stream0)
        buf377 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_133], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf374, buf376, buf375, primals_271, buf377, 20480, grid=grid(20480), stream=stream0)
        del buf375
        del primals_271
        # Topologically Sorted Source Nodes: [input_277], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_272, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf379 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [input_277, mul_34, input_278, mul_35, input_279], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf379, primals_273, buf365, buf337, 4096, grid=grid(4096), stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [input_280], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_274, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf381 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_280, input_281], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf380, primals_275, buf381, 4096, grid=grid(4096), stream=stream0)
        buf382 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_135], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf379, buf381, buf380, primals_275, buf382, 8192, grid=grid(8192), stream=stream0)
        del buf380
        del primals_275
        # Topologically Sorted Source Nodes: [input_282], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf384 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_282, input_283], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf383, primals_277, buf384, 4096, grid=grid(4096), stream=stream0)
        buf385 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_136], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf382, buf384, buf383, primals_277, buf385, 12288, grid=grid(12288), stream=stream0)
        del buf383
        del primals_277
        # Topologically Sorted Source Nodes: [input_284], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_278, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf387 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_284, input_285], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf386, primals_279, buf387, 4096, grid=grid(4096), stream=stream0)
        buf388 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_137], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf385, buf387, buf386, primals_279, buf388, 16384, grid=grid(16384), stream=stream0)
        del buf386
        del primals_279
        # Topologically Sorted Source Nodes: [input_286], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_280, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf390 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_286, input_287], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf389, primals_281, buf390, 4096, grid=grid(4096), stream=stream0)
        buf391 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_138], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf388, buf390, buf389, primals_281, buf391, 20480, grid=grid(20480), stream=stream0)
        del buf389
        del primals_281
        # Topologically Sorted Source Nodes: [input_288], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf393 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [input_288, mul_36, input_289], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf393, primals_283, buf379, 4096, grid=grid(4096), stream=stream0)
        del primals_283
        # Topologically Sorted Source Nodes: [input_290], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_284, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf395 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_290, input_291], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf394, primals_285, buf395, 4096, grid=grid(4096), stream=stream0)
        buf396 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_140], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf393, buf395, buf394, primals_285, buf396, 8192, grid=grid(8192), stream=stream0)
        del buf394
        del primals_285
        # Topologically Sorted Source Nodes: [input_292], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, primals_286, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf398 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_292, input_293], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf397, primals_287, buf398, 4096, grid=grid(4096), stream=stream0)
        buf399 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_141], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf396, buf398, buf397, primals_287, buf399, 12288, grid=grid(12288), stream=stream0)
        del buf397
        del primals_287
        # Topologically Sorted Source Nodes: [input_294], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, primals_288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf401 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_294, input_295], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf400, primals_289, buf401, 4096, grid=grid(4096), stream=stream0)
        buf402 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_142], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf399, buf401, buf400, primals_289, buf402, 16384, grid=grid(16384), stream=stream0)
        del buf400
        del primals_289
        # Topologically Sorted Source Nodes: [input_296], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_290, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf404 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_296, input_297], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf403, primals_291, buf404, 4096, grid=grid(4096), stream=stream0)
        buf405 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_143], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf402, buf404, buf403, primals_291, buf405, 20480, grid=grid(20480), stream=stream0)
        del buf403
        del primals_291
        # Topologically Sorted Source Nodes: [input_298], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf407 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [input_298, mul_37, input_299], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf407, primals_293, buf393, 4096, grid=grid(4096), stream=stream0)
        del primals_293
        # Topologically Sorted Source Nodes: [input_300], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, primals_294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf409 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_300, input_301], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf408, primals_295, buf409, 4096, grid=grid(4096), stream=stream0)
        buf410 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_145], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf407, buf409, buf408, primals_295, buf410, 8192, grid=grid(8192), stream=stream0)
        del buf408
        del primals_295
        # Topologically Sorted Source Nodes: [input_302], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_296, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf412 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_302, input_303], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf411, primals_297, buf412, 4096, grid=grid(4096), stream=stream0)
        buf413 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_146], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf410, buf412, buf411, primals_297, buf413, 12288, grid=grid(12288), stream=stream0)
        del buf411
        del primals_297
        # Topologically Sorted Source Nodes: [input_304], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_298, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf415 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_304, input_305], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf414, primals_299, buf415, 4096, grid=grid(4096), stream=stream0)
        buf416 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_147], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf413, buf415, buf414, primals_299, buf416, 16384, grid=grid(16384), stream=stream0)
        del buf414
        del primals_299
        # Topologically Sorted Source Nodes: [input_306], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, primals_300, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf418 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_306, input_307], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf417, primals_301, buf418, 4096, grid=grid(4096), stream=stream0)
        buf419 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_148], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf416, buf418, buf417, primals_301, buf419, 20480, grid=grid(20480), stream=stream0)
        del buf417
        del primals_301
        # Topologically Sorted Source Nodes: [input_308], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf419, primals_302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf421 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [input_308, mul_38, input_309, mul_39, input_310], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf421, primals_303, buf407, buf379, 4096, grid=grid(4096), stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [input_311], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_304, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf423 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_311, input_312], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf422, primals_305, buf423, 4096, grid=grid(4096), stream=stream0)
        buf424 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_150], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf421, buf423, buf422, primals_305, buf424, 8192, grid=grid(8192), stream=stream0)
        del buf422
        del primals_305
        # Topologically Sorted Source Nodes: [input_313], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf426 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_313, input_314], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf425, primals_307, buf426, 4096, grid=grid(4096), stream=stream0)
        buf427 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_151], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf424, buf426, buf425, primals_307, buf427, 12288, grid=grid(12288), stream=stream0)
        del buf425
        del primals_307
        # Topologically Sorted Source Nodes: [input_315], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_308, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf429 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_315, input_316], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf428, primals_309, buf429, 4096, grid=grid(4096), stream=stream0)
        buf430 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_152], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf427, buf429, buf428, primals_309, buf430, 16384, grid=grid(16384), stream=stream0)
        del buf428
        del primals_309
        # Topologically Sorted Source Nodes: [input_317], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_310, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf432 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_317, input_318], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf431, primals_311, buf432, 4096, grid=grid(4096), stream=stream0)
        buf433 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_153], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf430, buf432, buf431, primals_311, buf433, 20480, grid=grid(20480), stream=stream0)
        del buf431
        del primals_311
        # Topologically Sorted Source Nodes: [input_319], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf435 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [input_319, mul_40, input_320], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf435, primals_313, buf421, 4096, grid=grid(4096), stream=stream0)
        del primals_313
        # Topologically Sorted Source Nodes: [input_321], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, primals_314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf437 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_321, input_322], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf436, primals_315, buf437, 4096, grid=grid(4096), stream=stream0)
        buf438 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_155], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf435, buf437, buf436, primals_315, buf438, 8192, grid=grid(8192), stream=stream0)
        del buf436
        del primals_315
        # Topologically Sorted Source Nodes: [input_323], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_316, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf440 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_323, input_324], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf439, primals_317, buf440, 4096, grid=grid(4096), stream=stream0)
        buf441 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_156], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf438, buf440, buf439, primals_317, buf441, 12288, grid=grid(12288), stream=stream0)
        del buf439
        del primals_317
        # Topologically Sorted Source Nodes: [input_325], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, primals_318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf443 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_325, input_326], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf442, primals_319, buf443, 4096, grid=grid(4096), stream=stream0)
        buf444 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_157], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf441, buf443, buf442, primals_319, buf444, 16384, grid=grid(16384), stream=stream0)
        del buf442
        del primals_319
        # Topologically Sorted Source Nodes: [input_327], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, primals_320, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf446 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_327, input_328], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf445, primals_321, buf446, 4096, grid=grid(4096), stream=stream0)
        buf447 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_158], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf444, buf446, buf445, primals_321, buf447, 20480, grid=grid(20480), stream=stream0)
        del buf445
        del primals_321
        # Topologically Sorted Source Nodes: [input_329], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf449 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [input_329, mul_41, input_330], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf449, primals_323, buf435, 4096, grid=grid(4096), stream=stream0)
        del primals_323
        # Topologically Sorted Source Nodes: [input_331], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, primals_324, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf451 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_331, input_332], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf450, primals_325, buf451, 4096, grid=grid(4096), stream=stream0)
        buf452 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_160], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf449, buf451, buf450, primals_325, buf452, 8192, grid=grid(8192), stream=stream0)
        del buf450
        del primals_325
        # Topologically Sorted Source Nodes: [input_333], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_326, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf454 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_333, input_334], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf453, primals_327, buf454, 4096, grid=grid(4096), stream=stream0)
        buf455 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_161], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf452, buf454, buf453, primals_327, buf455, 12288, grid=grid(12288), stream=stream0)
        del buf453
        del primals_327
        # Topologically Sorted Source Nodes: [input_335], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_328, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf457 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_335, input_336], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf456, primals_329, buf457, 4096, grid=grid(4096), stream=stream0)
        buf458 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_162], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf455, buf457, buf456, primals_329, buf458, 16384, grid=grid(16384), stream=stream0)
        del buf456
        del primals_329
        # Topologically Sorted Source Nodes: [input_337], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, primals_330, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf460 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_337, input_338], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf459, primals_331, buf460, 4096, grid=grid(4096), stream=stream0)
        buf461 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_163], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf458, buf460, buf459, primals_331, buf461, 20480, grid=grid(20480), stream=stream0)
        del buf459
        del primals_331
        # Topologically Sorted Source Nodes: [input_339], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf463 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [input_339, mul_42, input_340, mul_43, input_341], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf463, primals_333, buf449, buf421, 4096, grid=grid(4096), stream=stream0)
        del primals_333
        # Topologically Sorted Source Nodes: [input_342], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_334, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf465 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_342, input_343], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf464, primals_335, buf465, 4096, grid=grid(4096), stream=stream0)
        buf466 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_165], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf463, buf465, buf464, primals_335, buf466, 8192, grid=grid(8192), stream=stream0)
        del buf464
        del primals_335
        # Topologically Sorted Source Nodes: [input_344], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_336, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf468 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_344, input_345], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf467, primals_337, buf468, 4096, grid=grid(4096), stream=stream0)
        buf469 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_166], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf466, buf468, buf467, primals_337, buf469, 12288, grid=grid(12288), stream=stream0)
        del buf467
        del primals_337
        # Topologically Sorted Source Nodes: [input_346], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_338, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf471 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_346, input_347], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf470, primals_339, buf471, 4096, grid=grid(4096), stream=stream0)
        buf472 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_167], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf469, buf471, buf470, primals_339, buf472, 16384, grid=grid(16384), stream=stream0)
        del buf470
        del primals_339
        # Topologically Sorted Source Nodes: [input_348], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_340, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf474 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_348, input_349], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf473, primals_341, buf474, 4096, grid=grid(4096), stream=stream0)
        buf475 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_168], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf472, buf474, buf473, primals_341, buf475, 20480, grid=grid(20480), stream=stream0)
        del buf473
        del primals_341
        # Topologically Sorted Source Nodes: [input_350], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf477 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [input_350, mul_44, input_351], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf477, primals_343, buf463, 4096, grid=grid(4096), stream=stream0)
        del primals_343
        # Topologically Sorted Source Nodes: [input_352], Original ATen: [aten.convolution]
        buf478 = extern_kernels.convolution(buf477, primals_344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf478, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf479 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_352, input_353], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf478, primals_345, buf479, 4096, grid=grid(4096), stream=stream0)
        buf480 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_170], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf477, buf479, buf478, primals_345, buf480, 8192, grid=grid(8192), stream=stream0)
        del buf478
        del primals_345
        # Topologically Sorted Source Nodes: [input_354], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, primals_346, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf482 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_354, input_355], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf481, primals_347, buf482, 4096, grid=grid(4096), stream=stream0)
        buf483 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_171], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf480, buf482, buf481, primals_347, buf483, 12288, grid=grid(12288), stream=stream0)
        del buf481
        del primals_347
        # Topologically Sorted Source Nodes: [input_356], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_348, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf485 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_356, input_357], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf484, primals_349, buf485, 4096, grid=grid(4096), stream=stream0)
        buf486 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_172], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf483, buf485, buf484, primals_349, buf486, 16384, grid=grid(16384), stream=stream0)
        del buf484
        del primals_349
        # Topologically Sorted Source Nodes: [input_358], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, primals_350, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf488 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_358, input_359], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf487, primals_351, buf488, 4096, grid=grid(4096), stream=stream0)
        buf489 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_173], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf486, buf488, buf487, primals_351, buf489, 20480, grid=grid(20480), stream=stream0)
        del buf487
        del primals_351
        # Topologically Sorted Source Nodes: [input_360], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf491 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [input_360, mul_45, input_361], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf491, primals_353, buf477, 4096, grid=grid(4096), stream=stream0)
        del primals_353
        # Topologically Sorted Source Nodes: [input_362], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf491, primals_354, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf493 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_362, input_363], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf492, primals_355, buf493, 4096, grid=grid(4096), stream=stream0)
        buf494 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_175], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf491, buf493, buf492, primals_355, buf494, 8192, grid=grid(8192), stream=stream0)
        del buf492
        del primals_355
        # Topologically Sorted Source Nodes: [input_364], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_356, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf496 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_364, input_365], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf495, primals_357, buf496, 4096, grid=grid(4096), stream=stream0)
        buf497 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_176], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf494, buf496, buf495, primals_357, buf497, 12288, grid=grid(12288), stream=stream0)
        del buf495
        del primals_357
        # Topologically Sorted Source Nodes: [input_366], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_358, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf499 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_366, input_367], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf498, primals_359, buf499, 4096, grid=grid(4096), stream=stream0)
        buf500 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_177], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf497, buf499, buf498, primals_359, buf500, 16384, grid=grid(16384), stream=stream0)
        del buf498
        del primals_359
        # Topologically Sorted Source Nodes: [input_368], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf502 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_368, input_369], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf501, primals_361, buf502, 4096, grid=grid(4096), stream=stream0)
        buf503 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_178], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf500, buf502, buf501, primals_361, buf503, 20480, grid=grid(20480), stream=stream0)
        del buf501
        del primals_361
        # Topologically Sorted Source Nodes: [input_370], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, primals_362, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf505 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [input_370, mul_46, input_371, mul_47, input_372], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf505, primals_363, buf491, buf463, 4096, grid=grid(4096), stream=stream0)
        del primals_363
        # Topologically Sorted Source Nodes: [input_373], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_364, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf507 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_373, input_374], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf506, primals_365, buf507, 4096, grid=grid(4096), stream=stream0)
        buf508 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_180], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf505, buf507, buf506, primals_365, buf508, 8192, grid=grid(8192), stream=stream0)
        del buf506
        del primals_365
        # Topologically Sorted Source Nodes: [input_375], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, primals_366, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf510 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_375, input_376], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf509, primals_367, buf510, 4096, grid=grid(4096), stream=stream0)
        buf511 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_181], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf508, buf510, buf509, primals_367, buf511, 12288, grid=grid(12288), stream=stream0)
        del buf509
        del primals_367
        # Topologically Sorted Source Nodes: [input_377], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_368, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf513 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_377, input_378], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf512, primals_369, buf513, 4096, grid=grid(4096), stream=stream0)
        buf514 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_182], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf511, buf513, buf512, primals_369, buf514, 16384, grid=grid(16384), stream=stream0)
        del buf512
        del primals_369
        # Topologically Sorted Source Nodes: [input_379], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_370, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf516 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_379, input_380], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf515, primals_371, buf516, 4096, grid=grid(4096), stream=stream0)
        buf517 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_183], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf514, buf516, buf515, primals_371, buf517, 20480, grid=grid(20480), stream=stream0)
        del buf515
        del primals_371
        # Topologically Sorted Source Nodes: [input_381], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, primals_372, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf519 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [input_381, mul_48, input_382], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf519, primals_373, buf505, 4096, grid=grid(4096), stream=stream0)
        del primals_373
        # Topologically Sorted Source Nodes: [input_383], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, primals_374, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf521 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_383, input_384], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf520, primals_375, buf521, 4096, grid=grid(4096), stream=stream0)
        buf522 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_185], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf519, buf521, buf520, primals_375, buf522, 8192, grid=grid(8192), stream=stream0)
        del buf520
        del primals_375
        # Topologically Sorted Source Nodes: [input_385], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf524 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_385, input_386], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf523, primals_377, buf524, 4096, grid=grid(4096), stream=stream0)
        buf525 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_186], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf522, buf524, buf523, primals_377, buf525, 12288, grid=grid(12288), stream=stream0)
        del buf523
        del primals_377
        # Topologically Sorted Source Nodes: [input_387], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_378, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf527 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_387, input_388], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf526, primals_379, buf527, 4096, grid=grid(4096), stream=stream0)
        buf528 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_187], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf525, buf527, buf526, primals_379, buf528, 16384, grid=grid(16384), stream=stream0)
        del buf526
        del primals_379
        # Topologically Sorted Source Nodes: [input_389], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf528, primals_380, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf530 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_389, input_390], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf529, primals_381, buf530, 4096, grid=grid(4096), stream=stream0)
        buf531 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_188], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf528, buf530, buf529, primals_381, buf531, 20480, grid=grid(20480), stream=stream0)
        del buf529
        del primals_381
        # Topologically Sorted Source Nodes: [input_391], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf531, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf533 = buf532; del buf532  # reuse
        # Topologically Sorted Source Nodes: [input_391, mul_49, input_392], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf533, primals_383, buf519, 4096, grid=grid(4096), stream=stream0)
        del primals_383
        # Topologically Sorted Source Nodes: [input_393], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_384, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf535 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_393, input_394], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf534, primals_385, buf535, 4096, grid=grid(4096), stream=stream0)
        buf536 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_190], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf533, buf535, buf534, primals_385, buf536, 8192, grid=grid(8192), stream=stream0)
        del buf534
        del primals_385
        # Topologically Sorted Source Nodes: [input_395], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf536, primals_386, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf537, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf538 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_395, input_396], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf537, primals_387, buf538, 4096, grid=grid(4096), stream=stream0)
        buf539 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_191], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf536, buf538, buf537, primals_387, buf539, 12288, grid=grid(12288), stream=stream0)
        del buf537
        del primals_387
        # Topologically Sorted Source Nodes: [input_397], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, primals_388, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf541 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_397, input_398], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf540, primals_389, buf541, 4096, grid=grid(4096), stream=stream0)
        buf542 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_192], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf539, buf541, buf540, primals_389, buf542, 16384, grid=grid(16384), stream=stream0)
        del buf540
        del primals_389
        # Topologically Sorted Source Nodes: [input_399], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, primals_390, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf544 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_399, input_400], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf543, primals_391, buf544, 4096, grid=grid(4096), stream=stream0)
        buf545 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_193], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf542, buf544, buf543, primals_391, buf545, 20480, grid=grid(20480), stream=stream0)
        del buf543
        del primals_391
        # Topologically Sorted Source Nodes: [input_401], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, primals_392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf547 = buf546; del buf546  # reuse
        # Topologically Sorted Source Nodes: [input_401, mul_50, input_402, mul_51, input_403], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf547, primals_393, buf533, buf505, 4096, grid=grid(4096), stream=stream0)
        del primals_393
        # Topologically Sorted Source Nodes: [input_404], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, primals_394, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf549 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_404, input_405], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf548, primals_395, buf549, 4096, grid=grid(4096), stream=stream0)
        buf550 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_195], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf547, buf549, buf548, primals_395, buf550, 8192, grid=grid(8192), stream=stream0)
        del buf548
        del primals_395
        # Topologically Sorted Source Nodes: [input_406], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_396, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf552 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_406, input_407], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf551, primals_397, buf552, 4096, grid=grid(4096), stream=stream0)
        buf553 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_196], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf550, buf552, buf551, primals_397, buf553, 12288, grid=grid(12288), stream=stream0)
        del buf551
        del primals_397
        # Topologically Sorted Source Nodes: [input_408], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, primals_398, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf555 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_408, input_409], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf554, primals_399, buf555, 4096, grid=grid(4096), stream=stream0)
        buf556 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_197], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf553, buf555, buf554, primals_399, buf556, 16384, grid=grid(16384), stream=stream0)
        del buf554
        del primals_399
        # Topologically Sorted Source Nodes: [input_410], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, primals_400, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf558 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_410, input_411], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf557, primals_401, buf558, 4096, grid=grid(4096), stream=stream0)
        buf559 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_198], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf556, buf558, buf557, primals_401, buf559, 20480, grid=grid(20480), stream=stream0)
        del buf557
        del primals_401
        # Topologically Sorted Source Nodes: [input_412], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf561 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [input_412, mul_52, input_413], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf561, primals_403, buf547, 4096, grid=grid(4096), stream=stream0)
        del primals_403
        # Topologically Sorted Source Nodes: [input_414], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf561, primals_404, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf563 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_414, input_415], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf562, primals_405, buf563, 4096, grid=grid(4096), stream=stream0)
        buf564 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_200], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf561, buf563, buf562, primals_405, buf564, 8192, grid=grid(8192), stream=stream0)
        del buf562
        del primals_405
        # Topologically Sorted Source Nodes: [input_416], Original ATen: [aten.convolution]
        buf565 = extern_kernels.convolution(buf564, primals_406, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf565, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf566 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_416, input_417], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf565, primals_407, buf566, 4096, grid=grid(4096), stream=stream0)
        buf567 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_201], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf564, buf566, buf565, primals_407, buf567, 12288, grid=grid(12288), stream=stream0)
        del buf565
        del primals_407
        # Topologically Sorted Source Nodes: [input_418], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf567, primals_408, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf568, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf569 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_418, input_419], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf568, primals_409, buf569, 4096, grid=grid(4096), stream=stream0)
        buf570 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_202], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf567, buf569, buf568, primals_409, buf570, 16384, grid=grid(16384), stream=stream0)
        del buf568
        del primals_409
        # Topologically Sorted Source Nodes: [input_420], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, primals_410, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf572 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_420, input_421], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf571, primals_411, buf572, 4096, grid=grid(4096), stream=stream0)
        buf573 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_203], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf570, buf572, buf571, primals_411, buf573, 20480, grid=grid(20480), stream=stream0)
        del buf571
        del primals_411
        # Topologically Sorted Source Nodes: [input_422], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf575 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [input_422, mul_53, input_423], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf575, primals_413, buf561, 4096, grid=grid(4096), stream=stream0)
        del primals_413
        # Topologically Sorted Source Nodes: [input_424], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_414, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf577 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_424, input_425], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf576, primals_415, buf577, 4096, grid=grid(4096), stream=stream0)
        buf578 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_205], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf575, buf577, buf576, primals_415, buf578, 8192, grid=grid(8192), stream=stream0)
        del buf576
        del primals_415
        # Topologically Sorted Source Nodes: [input_426], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, primals_416, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf580 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_426, input_427], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf579, primals_417, buf580, 4096, grid=grid(4096), stream=stream0)
        buf581 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_206], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf578, buf580, buf579, primals_417, buf581, 12288, grid=grid(12288), stream=stream0)
        del buf579
        del primals_417
        # Topologically Sorted Source Nodes: [input_428], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, primals_418, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf583 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_428, input_429], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf582, primals_419, buf583, 4096, grid=grid(4096), stream=stream0)
        buf584 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_207], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf581, buf583, buf582, primals_419, buf584, 16384, grid=grid(16384), stream=stream0)
        del buf582
        del primals_419
        # Topologically Sorted Source Nodes: [input_430], Original ATen: [aten.convolution]
        buf585 = extern_kernels.convolution(buf584, primals_420, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf585, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf586 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_430, input_431], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf585, primals_421, buf586, 4096, grid=grid(4096), stream=stream0)
        buf587 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_208], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf584, buf586, buf585, primals_421, buf587, 20480, grid=grid(20480), stream=stream0)
        del buf585
        del primals_421
        # Topologically Sorted Source Nodes: [input_432], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf587, primals_422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf589 = buf588; del buf588  # reuse
        # Topologically Sorted Source Nodes: [input_432, mul_54, input_433, mul_55, input_434], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf589, primals_423, buf575, buf547, 4096, grid=grid(4096), stream=stream0)
        del primals_423
        # Topologically Sorted Source Nodes: [input_435], Original ATen: [aten.convolution]
        buf590 = extern_kernels.convolution(buf589, primals_424, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf590, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf591 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_435, input_436], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf590, primals_425, buf591, 4096, grid=grid(4096), stream=stream0)
        buf592 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_210], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf589, buf591, buf590, primals_425, buf592, 8192, grid=grid(8192), stream=stream0)
        del buf590
        del primals_425
        # Topologically Sorted Source Nodes: [input_437], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, primals_426, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf594 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_437, input_438], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf593, primals_427, buf594, 4096, grid=grid(4096), stream=stream0)
        buf595 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_211], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf592, buf594, buf593, primals_427, buf595, 12288, grid=grid(12288), stream=stream0)
        del buf593
        del primals_427
        # Topologically Sorted Source Nodes: [input_439], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_428, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf597 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_439, input_440], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf596, primals_429, buf597, 4096, grid=grid(4096), stream=stream0)
        buf598 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_212], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf595, buf597, buf596, primals_429, buf598, 16384, grid=grid(16384), stream=stream0)
        del buf596
        del primals_429
        # Topologically Sorted Source Nodes: [input_441], Original ATen: [aten.convolution]
        buf599 = extern_kernels.convolution(buf598, primals_430, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf599, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf600 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_441, input_442], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf599, primals_431, buf600, 4096, grid=grid(4096), stream=stream0)
        buf601 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_213], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf598, buf600, buf599, primals_431, buf601, 20480, grid=grid(20480), stream=stream0)
        del buf599
        del primals_431
        # Topologically Sorted Source Nodes: [input_443], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, primals_432, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf603 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [input_443, mul_56, input_444], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf603, primals_433, buf589, 4096, grid=grid(4096), stream=stream0)
        del primals_433
        # Topologically Sorted Source Nodes: [input_445], Original ATen: [aten.convolution]
        buf604 = extern_kernels.convolution(buf603, primals_434, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf604, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf605 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_445, input_446], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf604, primals_435, buf605, 4096, grid=grid(4096), stream=stream0)
        buf606 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_215], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf603, buf605, buf604, primals_435, buf606, 8192, grid=grid(8192), stream=stream0)
        del buf604
        del primals_435
        # Topologically Sorted Source Nodes: [input_447], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(buf606, primals_436, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf607, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf608 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_447, input_448], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf607, primals_437, buf608, 4096, grid=grid(4096), stream=stream0)
        buf609 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_216], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf606, buf608, buf607, primals_437, buf609, 12288, grid=grid(12288), stream=stream0)
        del buf607
        del primals_437
        # Topologically Sorted Source Nodes: [input_449], Original ATen: [aten.convolution]
        buf610 = extern_kernels.convolution(buf609, primals_438, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf610, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf611 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_449, input_450], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf610, primals_439, buf611, 4096, grid=grid(4096), stream=stream0)
        buf612 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_217], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf609, buf611, buf610, primals_439, buf612, 16384, grid=grid(16384), stream=stream0)
        del buf610
        del primals_439
        # Topologically Sorted Source Nodes: [input_451], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, primals_440, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf613, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf614 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_451, input_452], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf613, primals_441, buf614, 4096, grid=grid(4096), stream=stream0)
        buf615 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_218], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf612, buf614, buf613, primals_441, buf615, 20480, grid=grid(20480), stream=stream0)
        del buf613
        del primals_441
        # Topologically Sorted Source Nodes: [input_453], Original ATen: [aten.convolution]
        buf616 = extern_kernels.convolution(buf615, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf616, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf617 = buf616; del buf616  # reuse
        # Topologically Sorted Source Nodes: [input_453, mul_57, input_454], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf617, primals_443, buf603, 4096, grid=grid(4096), stream=stream0)
        del primals_443
        # Topologically Sorted Source Nodes: [input_455], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, primals_444, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf619 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_455, input_456], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf618, primals_445, buf619, 4096, grid=grid(4096), stream=stream0)
        buf620 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_220], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf617, buf619, buf618, primals_445, buf620, 8192, grid=grid(8192), stream=stream0)
        del buf618
        del primals_445
        # Topologically Sorted Source Nodes: [input_457], Original ATen: [aten.convolution]
        buf621 = extern_kernels.convolution(buf620, primals_446, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf621, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf622 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_457, input_458], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf621, primals_447, buf622, 4096, grid=grid(4096), stream=stream0)
        buf623 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_221], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf620, buf622, buf621, primals_447, buf623, 12288, grid=grid(12288), stream=stream0)
        del buf621
        del primals_447
        # Topologically Sorted Source Nodes: [input_459], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf623, primals_448, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf625 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_459, input_460], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf624, primals_449, buf625, 4096, grid=grid(4096), stream=stream0)
        buf626 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_222], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf623, buf625, buf624, primals_449, buf626, 16384, grid=grid(16384), stream=stream0)
        del buf624
        del primals_449
        # Topologically Sorted Source Nodes: [input_461], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(buf626, primals_450, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf628 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_461, input_462], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf627, primals_451, buf628, 4096, grid=grid(4096), stream=stream0)
        buf629 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_223], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf626, buf628, buf627, primals_451, buf629, 20480, grid=grid(20480), stream=stream0)
        del buf627
        del primals_451
        # Topologically Sorted Source Nodes: [input_463], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, primals_452, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf631 = buf630; del buf630  # reuse
        # Topologically Sorted Source Nodes: [input_463, mul_58, input_464, mul_59, input_465], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf631, primals_453, buf617, buf589, 4096, grid=grid(4096), stream=stream0)
        del primals_453
        # Topologically Sorted Source Nodes: [input_466], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_454, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf633 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_466, input_467], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf632, primals_455, buf633, 4096, grid=grid(4096), stream=stream0)
        buf634 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_225], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf631, buf633, buf632, primals_455, buf634, 8192, grid=grid(8192), stream=stream0)
        del buf632
        del primals_455
        # Topologically Sorted Source Nodes: [input_468], Original ATen: [aten.convolution]
        buf635 = extern_kernels.convolution(buf634, primals_456, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf635, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf636 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_468, input_469], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf635, primals_457, buf636, 4096, grid=grid(4096), stream=stream0)
        buf637 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_226], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf634, buf636, buf635, primals_457, buf637, 12288, grid=grid(12288), stream=stream0)
        del buf635
        del primals_457
        # Topologically Sorted Source Nodes: [input_470], Original ATen: [aten.convolution]
        buf638 = extern_kernels.convolution(buf637, primals_458, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf638, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf639 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_470, input_471], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf638, primals_459, buf639, 4096, grid=grid(4096), stream=stream0)
        buf640 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_227], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf637, buf639, buf638, primals_459, buf640, 16384, grid=grid(16384), stream=stream0)
        del buf638
        del primals_459
        # Topologically Sorted Source Nodes: [input_472], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, primals_460, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf642 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_472, input_473], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf641, primals_461, buf642, 4096, grid=grid(4096), stream=stream0)
        buf643 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_228], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf640, buf642, buf641, primals_461, buf643, 20480, grid=grid(20480), stream=stream0)
        del buf641
        del primals_461
        # Topologically Sorted Source Nodes: [input_474], Original ATen: [aten.convolution]
        buf644 = extern_kernels.convolution(buf643, primals_462, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf644, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf645 = buf644; del buf644  # reuse
        # Topologically Sorted Source Nodes: [input_474, mul_60, input_475], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf645, primals_463, buf631, 4096, grid=grid(4096), stream=stream0)
        del primals_463
        # Topologically Sorted Source Nodes: [input_476], Original ATen: [aten.convolution]
        buf646 = extern_kernels.convolution(buf645, primals_464, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf647 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_476, input_477], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf646, primals_465, buf647, 4096, grid=grid(4096), stream=stream0)
        buf648 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_230], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf645, buf647, buf646, primals_465, buf648, 8192, grid=grid(8192), stream=stream0)
        del buf646
        del primals_465
        # Topologically Sorted Source Nodes: [input_478], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf648, primals_466, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf650 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_478, input_479], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf649, primals_467, buf650, 4096, grid=grid(4096), stream=stream0)
        buf651 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_231], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf648, buf650, buf649, primals_467, buf651, 12288, grid=grid(12288), stream=stream0)
        del buf649
        del primals_467
        # Topologically Sorted Source Nodes: [input_480], Original ATen: [aten.convolution]
        buf652 = extern_kernels.convolution(buf651, primals_468, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf652, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf653 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_480, input_481], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf652, primals_469, buf653, 4096, grid=grid(4096), stream=stream0)
        buf654 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_232], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf651, buf653, buf652, primals_469, buf654, 16384, grid=grid(16384), stream=stream0)
        del buf652
        del primals_469
        # Topologically Sorted Source Nodes: [input_482], Original ATen: [aten.convolution]
        buf655 = extern_kernels.convolution(buf654, primals_470, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf655, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf656 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_482, input_483], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf655, primals_471, buf656, 4096, grid=grid(4096), stream=stream0)
        buf657 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_233], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf654, buf656, buf655, primals_471, buf657, 20480, grid=grid(20480), stream=stream0)
        del buf655
        del primals_471
        # Topologically Sorted Source Nodes: [input_484], Original ATen: [aten.convolution]
        buf658 = extern_kernels.convolution(buf657, primals_472, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf658, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf659 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [input_484, mul_61, input_485], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_6.run(buf659, primals_473, buf645, 4096, grid=grid(4096), stream=stream0)
        del primals_473
        # Topologically Sorted Source Nodes: [input_486], Original ATen: [aten.convolution]
        buf660 = extern_kernels.convolution(buf659, primals_474, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf660, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf661 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_486, input_487], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf660, primals_475, buf661, 4096, grid=grid(4096), stream=stream0)
        buf662 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_235], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf659, buf661, buf660, primals_475, buf662, 8192, grid=grid(8192), stream=stream0)
        del buf660
        del primals_475
        # Topologically Sorted Source Nodes: [input_488], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, primals_476, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf664 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_488, input_489], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf663, primals_477, buf664, 4096, grid=grid(4096), stream=stream0)
        buf665 = empty_strided_cuda((4, 192, 4, 4), (3072, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_236], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf662, buf664, buf663, primals_477, buf665, 12288, grid=grid(12288), stream=stream0)
        del buf663
        del primals_477
        # Topologically Sorted Source Nodes: [input_490], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, primals_478, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf667 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_490, input_491], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf666, primals_479, buf667, 4096, grid=grid(4096), stream=stream0)
        buf668 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_237], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf665, buf667, buf666, primals_479, buf668, 16384, grid=grid(16384), stream=stream0)
        del buf666
        del primals_479
        # Topologically Sorted Source Nodes: [input_492], Original ATen: [aten.convolution]
        buf669 = extern_kernels.convolution(buf668, primals_480, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf669, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf670 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_492, input_493], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf669, primals_481, buf670, 4096, grid=grid(4096), stream=stream0)
        buf671 = empty_strided_cuda((4, 320, 4, 4), (5120, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_238], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf668, buf670, buf669, primals_481, buf671, 20480, grid=grid(20480), stream=stream0)
        del buf669
        del primals_481
        # Topologically Sorted Source Nodes: [input_494], Original ATen: [aten.convolution]
        buf672 = extern_kernels.convolution(buf671, primals_482, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf672, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf673 = buf672; del buf672  # reuse
        # Topologically Sorted Source Nodes: [input_494, mul_62, input_495, mul_63, input_496], Original ATen: [aten.convolution, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_7.run(buf673, primals_483, buf659, buf631, 4096, grid=grid(4096), stream=stream0)
        del primals_483
        # Topologically Sorted Source Nodes: [out2], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, primals_484, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf675 = buf674; del buf674  # reuse
        # Topologically Sorted Source Nodes: [out2, out], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_8.run(buf675, buf1, primals_485, 4096, grid=grid(4096), stream=stream0)
        del primals_485
        # Topologically Sorted Source Nodes: [input_497], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, primals_486, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf677 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.bool)
        buf678 = empty_strided_cuda((4, 64, 4, 2, 4, 2), (4096, 64, 16, 8, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_497, input_498, input_499], Original ATen: [aten.convolution, aten.leaky_relu, aten.pixel_shuffle]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_pixel_shuffle_9.run(buf676, primals_487, buf677, buf678, 1024, 16, grid=grid(1024, 16), stream=stream0)
        del buf676
        del primals_487
        # Topologically Sorted Source Nodes: [input_500], Original ATen: [aten.convolution]
        buf679 = extern_kernels.convolution(reinterpret_tensor(buf678, (4, 64, 8, 8), (4096, 64, 8, 1), 0), primals_488, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf679, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf680 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.bool)
        buf681 = empty_strided_cuda((4, 64, 8, 2, 8, 2), (16384, 256, 32, 16, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_500, input_501, input_502], Original ATen: [aten.convolution, aten.leaky_relu, aten.pixel_shuffle]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_pixel_shuffle_10.run(buf679, primals_489, buf680, buf681, 1024, 64, grid=grid(1024, 64), stream=stream0)
        del buf679
        del primals_489
        # Topologically Sorted Source Nodes: [input_503], Original ATen: [aten.convolution]
        buf682 = extern_kernels.convolution(reinterpret_tensor(buf681, (4, 64, 16, 16), (16384, 256, 16, 1), 0), primals_490, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf682, (4, 64, 16, 16), (16384, 256, 16, 1))
        buf683 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.bool)
        buf684 = buf682; del buf682  # reuse
        # Topologically Sorted Source Nodes: [input_503, input_504], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_11.run(buf684, primals_491, buf683, 65536, grid=grid(65536), stream=stream0)
        del primals_491
        # Topologically Sorted Source Nodes: [input_505], Original ATen: [aten.convolution]
        buf685 = extern_kernels.convolution(buf684, primals_492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf685, (4, 4, 16, 16), (1024, 256, 16, 1))
        buf686 = buf685; del buf685  # reuse
        # Topologically Sorted Source Nodes: [input_505], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf686, primals_493, 4096, grid=grid(4096), stream=stream0)
        del primals_493
    return (buf686, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, primals_232, primals_234, primals_236, primals_238, primals_240, primals_242, primals_244, primals_246, primals_248, primals_250, primals_252, primals_254, primals_256, primals_258, primals_260, primals_262, primals_264, primals_266, primals_268, primals_270, primals_272, primals_274, primals_276, primals_278, primals_280, primals_282, primals_284, primals_286, primals_288, primals_290, primals_292, primals_294, primals_296, primals_298, primals_300, primals_302, primals_304, primals_306, primals_308, primals_310, primals_312, primals_314, primals_316, primals_318, primals_320, primals_322, primals_324, primals_326, primals_328, primals_330, primals_332, primals_334, primals_336, primals_338, primals_340, primals_342, primals_344, primals_346, primals_348, primals_350, primals_352, primals_354, primals_356, primals_358, primals_360, primals_362, primals_364, primals_366, primals_368, primals_370, primals_372, primals_374, primals_376, primals_378, primals_380, primals_382, primals_384, primals_386, primals_388, primals_390, primals_392, primals_394, primals_396, primals_398, primals_400, primals_402, primals_404, primals_406, primals_408, primals_410, primals_412, primals_414, primals_416, primals_418, primals_420, primals_422, primals_424, primals_426, primals_428, primals_430, primals_432, primals_434, primals_436, primals_438, primals_440, primals_442, primals_444, primals_446, primals_448, primals_450, primals_452, primals_454, primals_456, primals_458, primals_460, primals_462, primals_464, primals_466, primals_468, primals_470, primals_472, primals_474, primals_476, primals_478, primals_480, primals_482, primals_484, primals_486, primals_488, primals_490, primals_492, buf1, buf3, buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf15, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf29, buf31, buf32, buf34, buf35, buf37, buf38, buf40, buf41, buf43, buf45, buf46, buf48, buf49, buf51, buf52, buf54, buf55, buf57, buf59, buf60, buf62, buf63, buf65, buf66, buf68, buf69, buf71, buf73, buf74, buf76, buf77, buf79, buf80, buf82, buf83, buf85, buf87, buf88, buf90, buf91, buf93, buf94, buf96, buf97, buf99, buf101, buf102, buf104, buf105, buf107, buf108, buf110, buf111, buf113, buf115, buf116, buf118, buf119, buf121, buf122, buf124, buf125, buf127, buf129, buf130, buf132, buf133, buf135, buf136, buf138, buf139, buf141, buf143, buf144, buf146, buf147, buf149, buf150, buf152, buf153, buf155, buf157, buf158, buf160, buf161, buf163, buf164, buf166, buf167, buf169, buf171, buf172, buf174, buf175, buf177, buf178, buf180, buf181, buf183, buf185, buf186, buf188, buf189, buf191, buf192, buf194, buf195, buf197, buf199, buf200, buf202, buf203, buf205, buf206, buf208, buf209, buf211, buf213, buf214, buf216, buf217, buf219, buf220, buf222, buf223, buf225, buf227, buf228, buf230, buf231, buf233, buf234, buf236, buf237, buf239, buf241, buf242, buf244, buf245, buf247, buf248, buf250, buf251, buf253, buf255, buf256, buf258, buf259, buf261, buf262, buf264, buf265, buf267, buf269, buf270, buf272, buf273, buf275, buf276, buf278, buf279, buf281, buf283, buf284, buf286, buf287, buf289, buf290, buf292, buf293, buf295, buf297, buf298, buf300, buf301, buf303, buf304, buf306, buf307, buf309, buf311, buf312, buf314, buf315, buf317, buf318, buf320, buf321, buf323, buf325, buf326, buf328, buf329, buf331, buf332, buf334, buf335, buf337, buf339, buf340, buf342, buf343, buf345, buf346, buf348, buf349, buf351, buf353, buf354, buf356, buf357, buf359, buf360, buf362, buf363, buf365, buf367, buf368, buf370, buf371, buf373, buf374, buf376, buf377, buf379, buf381, buf382, buf384, buf385, buf387, buf388, buf390, buf391, buf393, buf395, buf396, buf398, buf399, buf401, buf402, buf404, buf405, buf407, buf409, buf410, buf412, buf413, buf415, buf416, buf418, buf419, buf421, buf423, buf424, buf426, buf427, buf429, buf430, buf432, buf433, buf435, buf437, buf438, buf440, buf441, buf443, buf444, buf446, buf447, buf449, buf451, buf452, buf454, buf455, buf457, buf458, buf460, buf461, buf463, buf465, buf466, buf468, buf469, buf471, buf472, buf474, buf475, buf477, buf479, buf480, buf482, buf483, buf485, buf486, buf488, buf489, buf491, buf493, buf494, buf496, buf497, buf499, buf500, buf502, buf503, buf505, buf507, buf508, buf510, buf511, buf513, buf514, buf516, buf517, buf519, buf521, buf522, buf524, buf525, buf527, buf528, buf530, buf531, buf533, buf535, buf536, buf538, buf539, buf541, buf542, buf544, buf545, buf547, buf549, buf550, buf552, buf553, buf555, buf556, buf558, buf559, buf561, buf563, buf564, buf566, buf567, buf569, buf570, buf572, buf573, buf575, buf577, buf578, buf580, buf581, buf583, buf584, buf586, buf587, buf589, buf591, buf592, buf594, buf595, buf597, buf598, buf600, buf601, buf603, buf605, buf606, buf608, buf609, buf611, buf612, buf614, buf615, buf617, buf619, buf620, buf622, buf623, buf625, buf626, buf628, buf629, buf631, buf633, buf634, buf636, buf637, buf639, buf640, buf642, buf643, buf645, buf647, buf648, buf650, buf651, buf653, buf654, buf656, buf657, buf659, buf661, buf662, buf664, buf665, buf667, buf668, buf670, buf671, buf673, buf675, buf677, reinterpret_tensor(buf678, (4, 64, 8, 8), (4096, 64, 8, 1), 0), buf680, reinterpret_tensor(buf681, (4, 64, 16, 16), (16384, 256, 16, 1), 0), buf683, buf684, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((64, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((64, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((64, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((4, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
