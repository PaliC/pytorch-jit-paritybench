# AOT ID: ['63_forward']
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


# kernel path: inductor_cache/ir/cirxqee47y5pmkc5n54okqlvzkciybuxvw2dgwkscyolber2nvoq.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1 + 8*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xz/cxzldy2imvlfnqzzktlw4sjsaigpilw6muucr57rhi36tt55j3ol.py
# Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2*x1 + 8*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ut/cutdmchl7kt2gz6hmmqsq4dfj3kwctwc5p5sbxvenreywwtni7gq.py
# Topologically Sorted Source Nodes: [sigmoid, real_1], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   real_1 => mul
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_9,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %sigmoid), kwargs = {})
triton_poi_fused_mul_sigmoid_2 = async_compile.triton('triton_poi_fused_mul_sigmoid_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aw/cawsdvzimevtpcfctxzntxf6iu4ubvvp242v52q76pbbojbafs7z.py
# Topologically Sorted Source Nodes: [time_step_as_inner, view_as_complex], Original ATen: [aten.cat, aten.view_as_complex]
# Source node to ATen node mapping:
#   time_step_as_inner => cat
#   view_as_complex => view_as_complex
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_2, %unsqueeze_3], -1), kwargs = {})
#   %view_as_complex : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%cat,), kwargs = {})
triton_poi_fused_cat_view_as_complex_3 = async_compile.triton('triton_poi_fused_cat_view_as_complex_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_view_as_complex_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_view_as_complex_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x5 = xindex // 2
    x2 = ((xindex // 8) % 4)
    x3 = ((xindex // 32) % 4)
    x4 = xindex // 128
    x6 = (xindex % 8)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x5), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x5), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x5), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (x5), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 * tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x6 + 8*x3 + 32*x2 + 128*x4), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g5/cg556xx3kaod5jnm7n42lf35xowwode3qbvinwuhpsgedvdnwfc6.py
# Topologically Sorted Source Nodes: [igfted], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   igfted => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 64*y1), xmask & ymask)
    tl.store(out_ptr0 + (x2 + 16*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/mc/cmc77bci7rglakgwmd7st534e5clieplstxsg4cpfxq2kzvwtiie.py
# Topologically Sorted Source Nodes: [igfted], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   igfted => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hl/chlbxhdmrfwomoiv6a7roynwevpwkqcxj53opd2nd2ykhmngb7hn.py
# Topologically Sorted Source Nodes: [igfted_1], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   igfted_1 => sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_34, [1]), kwargs = {})
triton_poi_fused_sum_6 = async_compile.triton('triton_poi_fused_sum_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nu/cnubyegbkjl4r3jnxz2beprtbih3mroa3dza6yceqtxq7omwscix.py
# Topologically Sorted Source Nodes: [forecast_source], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   forecast_source => sigmoid_6
# Graph fragment:
#   %sigmoid_6 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%squeeze,), kwargs = {})
triton_poi_fused_sigmoid_7 = async_compile.triton('triton_poi_fused_sigmoid_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sigmoid_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vs/cvsmxfkfci6qgfflogljif5gmek3lmydecataji5pgbrbd4un2fn.py
# Topologically Sorted Source Nodes: [sub, backcast_source], Original ATen: [aten.sub, aten.sigmoid]
# Source node to ATen node mapping:
#   backcast_source => sigmoid_7
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_42, %squeeze_1), kwargs = {})
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%sub,), kwargs = {})
triton_poi_fused_sigmoid_sub_8 = async_compile.triton('triton_poi_fused_sigmoid_sub_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_sub_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sigmoid_sub_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 64
    x4 = (xindex % 16)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x4 + 16*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = tl.sigmoid(tmp6)
    tl.store(in_out_ptr0 + (x5), tmp7, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (16, 16), (16, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, 16), (16, 1))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, 16), (16, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, 16), (16, 1))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (16, 16), (16, 1))
    assert_size_stride(primals_12, (16, ), (1, ))
    assert_size_stride(primals_13, (16, 16), (16, 1))
    assert_size_stride(primals_14, (16, ), (1, ))
    assert_size_stride(primals_15, (16, 16), (16, 1))
    assert_size_stride(primals_16, (16, ), (1, ))
    assert_size_stride(primals_17, (16, 16), (16, 1))
    assert_size_stride(primals_18, (16, ), (1, ))
    assert_size_stride(primals_19, (16, 16), (16, 1))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (16, 16), (16, 1))
    assert_size_stride(primals_22, (16, ), (1, ))
    assert_size_stride(primals_23, (16, 16), (16, 1))
    assert_size_stride(primals_24, (16, ), (1, ))
    assert_size_stride(primals_25, (16, 16), (16, 1))
    assert_size_stride(primals_26, (16, ), (1, ))
    assert_size_stride(primals_27, (1, 4, 1, 4, 4), (64, 16, 16, 4, 1))
    assert_size_stride(primals_28, (4, 4), (4, 1))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, 4), (4, 1))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, 4), (4, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, 4), (4, 1))
    assert_size_stride(primals_35, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [gfted], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_1, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(primals_2, (16, 4, 4), (16, 4, 1), 0), out=buf0)
        del primals_1
        # Topologically Sorted Source Nodes: [fft_fft], Original ATen: [aten._fft_r2c]
        buf1 = torch.ops.aten._fft_r2c.default(reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0), [1], 0, False)
        buf2 = buf1
        del buf1
        # Topologically Sorted Source Nodes: [ffted], Original ATen: [aten.view_as_real]
        buf3 = torch.ops.aten.view_as_real.default(buf2)
        buf4 = buf3
        buf5 = reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf4, buf5, 256, grid=grid(256), stream=stream0)
        buf6 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_4, reinterpret_tensor(buf5, (16, 16), (16, 1), 0), reinterpret_tensor(primals_3, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf6)
        del primals_3
        del primals_4
        buf7 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf5, (16, 16), (16, 1), 0), reinterpret_tensor(primals_5, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf7)
        del primals_5
        del primals_6
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf4, buf8, 256, grid=grid(256), stream=stream0)
        del buf2
        del buf3
        del buf4
        buf9 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, reinterpret_tensor(buf8, (16, 16), (16, 1), 0), reinterpret_tensor(primals_7, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf9)
        del primals_7
        del primals_8
        buf10 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf8, (16, 16), (16, 1), 0), reinterpret_tensor(primals_9, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf10)
        del primals_10
        del primals_9
        buf11 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid, real_1], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_2.run(buf6, buf7, buf11, 256, grid=grid(256), stream=stream0)
        buf12 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, reinterpret_tensor(buf11, (16, 16), (16, 1), 0), reinterpret_tensor(primals_11, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf12)
        del primals_12
        buf13 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, reinterpret_tensor(buf11, (16, 16), (16, 1), 0), reinterpret_tensor(primals_13, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf13)
        del primals_14
        buf14 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_1, img_1], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_2.run(buf9, buf10, buf14, 256, grid=grid(256), stream=stream0)
        buf15 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_16, reinterpret_tensor(buf14, (16, 16), (16, 1), 0), reinterpret_tensor(primals_15, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf15)
        del primals_16
        buf16 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_18, reinterpret_tensor(buf14, (16, 16), (16, 1), 0), reinterpret_tensor(primals_17, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf16)
        del primals_18
        buf17 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_2, real_2], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_2.run(buf12, buf13, buf17, 256, grid=grid(256), stream=stream0)
        buf18 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_20, reinterpret_tensor(buf17, (16, 16), (16, 1), 0), reinterpret_tensor(primals_19, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf18)
        del primals_20
        buf19 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_22, reinterpret_tensor(buf17, (16, 16), (16, 1), 0), reinterpret_tensor(primals_21, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf19)
        del primals_22
        buf20 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_3, img_2], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_2.run(buf15, buf16, buf20, 256, grid=grid(256), stream=stream0)
        buf21 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, reinterpret_tensor(buf20, (16, 16), (16, 1), 0), reinterpret_tensor(primals_23, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf21)
        del primals_24
        buf22 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_26, reinterpret_tensor(buf20, (16, 16), (16, 1), 0), reinterpret_tensor(primals_25, (16, 16), (1, 16), 0), alpha=1, beta=1, out=buf22)
        del primals_26
        buf23 = empty_strided_cuda((4, 4, 4, 4, 2), (128, 32, 8, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [time_step_as_inner, view_as_complex], Original ATen: [aten.cat, aten.view_as_complex]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_view_as_complex_3.run(buf18, buf19, buf21, buf22, buf23, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [time_step_as_inner, view_as_complex], Original ATen: [aten.cat, aten.view_as_complex]
        buf24 = torch.ops.aten.view_as_complex.default(buf23)
        buf25 = buf24
        # Topologically Sorted Source Nodes: [iffted], Original ATen: [aten.slice]
        buf26 = torch.ops.aten.slice.Tensor(buf25, 1, 0, 3)
        buf27 = buf26
        # Topologically Sorted Source Nodes: [iffted], Original ATen: [aten._fft_c2r]
        buf28 = torch.ops.aten._fft_c2r.default(buf27, [1], 2, 4)
        del buf23
        del buf24
        del buf25
        del buf26
        del buf27
        buf29 = buf28
        del buf28
        buf30 = empty_strided_cuda((4, 4, 1, 4, 4), (64, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [igfted], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf29, buf30, 16, 16, grid=grid(16, 16), stream=stream0)
        buf31 = reinterpret_tensor(buf29, (4, 4, 1, 4, 4), (64, 16, 16, 4, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [igfted], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(primals_27, buf31, 256, grid=grid(256), stream=stream0)
        del primals_27
        buf32 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [igfted], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf31, (16, 4, 4), (16, 4, 1), 0), out=buf32)
        buf33 = empty_strided_cuda((4, 1, 4, 4), (16, 1, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [igfted_1], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sum_6.run(buf32, buf33, 64, grid=grid(64), stream=stream0)
        buf34 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf33, (16, 4), (4, 1), 0), reinterpret_tensor(primals_28, (4, 4), (1, 4), 0), out=buf34)
        buf35 = reinterpret_tensor(buf34, (4, 4, 4), (16, 4, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [forecast_source], Original ATen: [aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_7.run(buf35, primals_29, 64, grid=grid(64), stream=stream0)
        del primals_29
        buf36 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [forecast], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_31, reinterpret_tensor(buf35, (16, 4), (4, 1), 0), reinterpret_tensor(primals_30, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf36)
        del primals_31
        buf37 = reinterpret_tensor(buf32, (64, 4), (4, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_2, (64, 4), (4, 1), 0), reinterpret_tensor(primals_32, (4, 4), (1, 4), 0), out=buf37)
        del primals_32
        buf38 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf33, (16, 4), (4, 1), 0), reinterpret_tensor(primals_34, (4, 4), (1, 4), 0), out=buf38)
        buf39 = reinterpret_tensor(buf37, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [sub, backcast_source], Original ATen: [aten.sub, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_sub_8.run(buf39, buf38, primals_35, primals_33, 256, grid=grid(256), stream=stream0)
        del buf38
        del primals_33
        del primals_35
    return (reinterpret_tensor(buf36, (4, 4, 4), (16, 4, 1), 0), buf39, reinterpret_tensor(buf5, (16, 16), (16, 1), 0), buf6, buf7, reinterpret_tensor(buf8, (16, 16), (16, 1), 0), buf9, buf10, reinterpret_tensor(buf11, (16, 16), (16, 1), 0), buf12, buf13, reinterpret_tensor(buf14, (16, 16), (16, 1), 0), buf15, buf16, reinterpret_tensor(buf17, (16, 16), (16, 1), 0), buf18, buf19, reinterpret_tensor(buf20, (16, 16), (16, 1), 0), buf21, buf22, reinterpret_tensor(buf33, (16, 4), (4, 1), 0), buf35, reinterpret_tensor(primals_2, (64, 4), (4, 1), 0), buf39, primals_34, primals_30, primals_28, reinterpret_tensor(buf30, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf31, (16, 4, 4), (16, 1, 4), 0), primals_25, primals_23, primals_21, primals_19, primals_17, primals_15, primals_13, primals_11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, 4, 1, 4, 4), (64, 16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
