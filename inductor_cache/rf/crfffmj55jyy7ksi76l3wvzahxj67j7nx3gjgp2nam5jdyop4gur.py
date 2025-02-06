# AOT ID: ['186_forward']
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


# kernel path: inductor_cache/oe/coex4j5hvxwxzgaqrrye7uldzbwwjenytvtp3ahh6trbs3mmazrc.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/24/c24lg55ux25yc232bb4tq6ncfrp3y6b5kqsjj75yyag2t2q6oqvx.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xc/cxckqvjjaxzyvm46rnwmmtbbkwixsufrwwh5nhdc6gm3jrtaecha.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lb/clbvcn2ovmhqrtkk5antxevqrobjxs2f7aspjgknsyj34u56qcih.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 48)
    y1 = yindex // 48
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 432*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j3/cj3aqi2t7tn2uazy6pjodhxzr2ba6kojtzy2vgz6pgsu23idndl2.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/22/c22pbg64esmaueu4aehhxuyi5sex4lraxlchjxsjpjacls5yns3w.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 65536*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 196608*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/4p/c4poqdfyg4cxunzvo5zrixbnd7geu7qjx6oaoipvpjrpap2zq6s2.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => clamp_max, clamp_min
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/mq/cmqo3ejqjtccqjqtde4bqtdm6cplee3srcw3do5r2kiv6a4szaeh.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_4 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_7 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 63)
    x2 = ((xindex // 6048) % 63)
    x3 = xindex // 381024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (96 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (192 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (12288 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (12384 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (12480 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (24576 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (24672 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (24768 + x0 + 192*x1 + 24576*x2 + 1572864*x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sj/csjtbdvsyuwxf42g7dkbvri4gd4nnq5wgawhjpf75gann3vu52kc.py
# Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_24 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_1, 0.0), kwargs = {})
#   %ge_23 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_1, 6.0), kwargs = {})
#   %bitwise_or_23 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_24, %ge_23), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_8 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 254016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/is/cisrinab4cpnpt3pzvmash7bvxaoef3tdo5yztn5dmlx66qkeuch.py
# Topologically Sorted Source Nodes: [conv2d_1, x], Original ATen: [aten.convolution, aten.hardtanh]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   x => clamp_max_1, clamp_min_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_1, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6.0), kwargs = {})
triton_poi_fused_convolution_hardtanh_9 = async_compile.triton('triton_poi_fused_convolution_hardtanh_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 254016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qq/cqqzac2ykhvnrngllaiatvis2lkuvaivucgp2awarhu7rbnmh3kw.py
# Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_2 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_1, %primals_10, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_23 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_2, 0.0), kwargs = {})
#   %ge_22 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_2, 6.0), kwargs = {})
#   %bitwise_or_22 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_23, %ge_22), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_10 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1016064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dv/cdvdk47tpg6g5nlp4k5ruccvusgrqzinmsw67jfuryphadcesvwq.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_2, %clamp_max_3], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2032128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = 6.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 128, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr2 + (64*x1 + ((-64) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + ((-64) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = 6.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5iefytutlwml6mq2z3pmd7hrr46r2hegtzw4xa5j3ayrlf3emit.py
# Topologically Sorted Source Nodes: [conv2d_4, x_1], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_4 => convolution_4
#   x_1 => clamp_max_4, clamp_min_4
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_4, 0.0), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6.0), kwargs = {})
#   %le_21 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_4, 0.0), kwargs = {})
#   %ge_20 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_4, 6.0), kwargs = {})
#   %bitwise_or_20 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_21, %ge_20), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_12 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_12(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 254016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jr/cjraonly2fmtpwakhgswt3cl7aeuu55fiwbe2pqemugdgjb776o7.py
# Topologically Sorted Source Nodes: [conv2d_7, x_2], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_7 => convolution_7
#   x_2 => clamp_max_7, clamp_min_7
# Graph fragment:
#   %convolution_7 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_20, %primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_7, 0.0), kwargs = {})
#   %clamp_max_7 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 6.0), kwargs = {})
#   %le_18 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_7, 0.0), kwargs = {})
#   %ge_17 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_7, 6.0), kwargs = {})
#   %bitwise_or_17 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_18, %ge_17), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_13 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_13(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 508032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ii/ciirvvmqvw4ovdvkie5perxonhjdxlwx2645rlc5kaq3grzzqm7n.py
# Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_8 => convolution_8
# Graph fragment:
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_7, %primals_22, %primals_23, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_17 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_8, 0.0), kwargs = {})
#   %ge_16 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_8, 6.0), kwargs = {})
#   %bitwise_or_16 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_17, %ge_16), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_14 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2032128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvuzhxhqnvmdctk33eqxzgtz2eruowadzax6fckwzjrx26grmkd3.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_8, %clamp_max_9], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4064256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = 6.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 256, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr2 + (128*x1 + ((-128) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + ((-128) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = 6.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a3/ca36frmdo7rtfevhmmnzxnmo5cobhboapzeif5koqij4h5pr3dwz.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_5 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_16 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_16(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 984064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 31)
    x2 = ((xindex // 7936) % 31)
    x3 = xindex // 246016
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (16128 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (16384 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (16640 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (32256 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (32512 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (32768 + x0 + 512*x1 + 32256*x2 + 1016064*x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxhtjxjwzxgavr6vw62gaqmdfvmskbsas37ym2ylq4u2ciie2vt.py
# Topologically Sorted Source Nodes: [conv2d_10, x_3], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_10 => convolution_10
#   x_3 => clamp_max_10, clamp_min_10
# Graph fragment:
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_26, %primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_10, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6.0), kwargs = {})
#   %le_15 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_10, 0.0), kwargs = {})
#   %ge_14 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_10, 6.0), kwargs = {})
#   %bitwise_or_14 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_15, %ge_14), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_17 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_17(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 123008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/os/costvplwje2afzp2ypb75h67qqu32dt7diqere5p2tlyitlru4wh.py
# Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_11 => convolution_11
# Graph fragment:
#   %convolution_11 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_10, %primals_28, %primals_29, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_14 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_11, 0.0), kwargs = {})
#   %ge_13 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_11, 6.0), kwargs = {})
#   %bitwise_or_13 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_14, %ge_13), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_18 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 492032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x5/cx5o2yw4pg7ahulsq5b7rcc5abbzgxymruznxzdtzffwneucj3t5.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_11, %clamp_max_12], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 984064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = 6.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 256, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr2 + (128*x1 + ((-128) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + ((-128) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = 6.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5sjp3lus2vsj5qhpsem3wxxuygupvfut25suarcgmxf62oxfq5a.py
# Topologically Sorted Source Nodes: [conv2d_13, x_4], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_13 => convolution_13
#   x_4 => clamp_max_13, clamp_min_13
# Graph fragment:
#   %convolution_13 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_32, %primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_13, 0.0), kwargs = {})
#   %clamp_max_13 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_13, 6.0), kwargs = {})
#   %le_12 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_13, 0.0), kwargs = {})
#   %ge_11 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_13, 6.0), kwargs = {})
#   %bitwise_or_11 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_12, %ge_11), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_20 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_20(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 184512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7x/c7xmg5vqe7gjsdykmwqdx56zidwcmsyhbuzmtlrczfdgcbzlj3r6.py
# Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_14 => convolution_14
# Graph fragment:
#   %convolution_14 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_13, %primals_34, %primals_35, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_11 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_14, 0.0), kwargs = {})
#   %ge_10 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_14, 6.0), kwargs = {})
#   %bitwise_or_10 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_11, %ge_10), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_21 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 738048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cddjqcq5g4opibhvzhcdr6hki5wobonil2kgj5pjfb36a3vz6vx6.py
# Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_4 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_14, %clamp_max_15], 1), kwargs = {})
triton_poi_fused_cat_22 = async_compile.triton('triton_poi_fused_cat_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1476096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = xindex // 384
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (192*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = 6.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 384, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr2 + (192*x1 + ((-192) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + ((-192) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = 6.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oz/cozsy252ohkx7qhonbkvitpaqydprjbbt7hahnczudxfx43dyhda.py
# Topologically Sorted Source Nodes: [conv2d_19, x_6], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_19 => convolution_19
#   x_6 => clamp_max_19, clamp_min_19
# Graph fragment:
#   %convolution_19 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_5, %primals_44, %primals_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_19, 0.0), kwargs = {})
#   %clamp_max_19 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 6.0), kwargs = {})
#   %le_6 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_19, 0.0), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_19, 6.0), kwargs = {})
#   %bitwise_or_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_6, %ge_5), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_23 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_23(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 246016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lu/clufkdg23amb5oduxqygosf5a3f6gkqjcw5oypcquvsd266l4iom.py
# Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_20 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_19, %primals_46, %primals_47, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_5 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_20, 0.0), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_20, 6.0), kwargs = {})
#   %bitwise_or_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_5, %ge_4), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_24 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_24(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 984064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4b/c4bwimol7wis3zpa7ja3lc73rh5qomq577xa37b7c7dqpp3rnfdd.py
# Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_6 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_20, %clamp_max_21], 1), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1968128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = 6.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 512, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr2 + (256*x1 + ((-256) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + ((-256) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = 6.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/42/c42it626aq7veubrgxvsh2plqlazhaw24qvdcsl3muac7ewmx7od.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_6 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_26 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_26(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 460800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 15)
    x2 = ((xindex // 7680) % 15)
    x3 = xindex // 115200
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (1024 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (15872 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (16384 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (16896 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (31744 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (32256 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (32768 + x0 + 1024*x1 + 31744*x2 + 492032*x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bj/cbjbfqhulbtezdfosm6mdn5uyat7flmhub4iw7zjtyk6vkcivl5d.py
# Topologically Sorted Source Nodes: [conv2d_22, x_7], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_22 => convolution_22
#   x_7 => clamp_max_22, clamp_min_22
# Graph fragment:
#   %convolution_22 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_4, %primals_50, %primals_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clamp_min_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convolution_22, 0.0), kwargs = {})
#   %clamp_max_22 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_22, 6.0), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_22, 0.0), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_22, 6.0), kwargs = {})
#   %bitwise_or_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_3, %ge_2), kwargs = {})
triton_poi_fused_convolution_hardtanh_hardtanh_backward_27 = async_compile.triton('triton_poi_fused_convolution_hardtanh_hardtanh_backward_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_hardtanh_backward_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_27(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp2 <= tmp3
    tmp8 = tmp2 >= tmp5
    tmp9 = tmp7 | tmp8
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nx/cnxvfvefeyuxnnrhwbvo2gvnvrupdooxjazu3yctg7qi422hwenf.py
# Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   conv2d_23 => convolution_23
# Graph fragment:
#   %convolution_23 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_22, %primals_52, %primals_53, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%convolution_23, 0.0), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convolution_23, 6.0), kwargs = {})
#   %bitwise_or_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_2, %ge_1), kwargs = {})
triton_poi_fused_convolution_hardtanh_backward_28 = async_compile.triton('triton_poi_fused_convolution_hardtanh_backward_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardtanh_backward_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardtanh_backward_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp5 = 6.0
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 | tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nj/cnjgafjrafvpjlxdb3ktc4jd7eanxue4uaamrltgrarmz5lac76s.py
# Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_7 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_23, %clamp_max_24], 1), kwargs = {})
triton_poi_fused_cat_29 = async_compile.triton('triton_poi_fused_cat_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 460800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = 6.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 512, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr2 + (256*x1 + ((-256) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + ((-256) + x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = 6.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xj/cxjjwwtdmdtmpmbdzfmv3lpzunp3nal4tjzcfalo45lakvmrvn47.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_8 => convolution_25
#   input_9 => relu
# Graph fragment:
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_7, %primals_56, %primals_57, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
triton_poi_fused_convolution_relu_30 = async_compile.triton('triton_poi_fused_convolution_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 900000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1000)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2z/c2z6bida65skle3v5v255b2w3okeb3nx2kxqyzfaavpg5uyl4m72.py
# Topologically Sorted Source Nodes: [view], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   view => view
# Graph fragment:
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%avg_pool2d, [1000, -1]), kwargs = {})
triton_poi_fused_view_31 = async_compile.triton('triton_poi_fused_view_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 36)
    x1 = xindex // 36
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1000*((x0 % 9)) + 9000*((x0 + 36*x1) // 9000) + ((((x0 + 36*x1) // 9) % 1000))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57 = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (96, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(primals_4, (96, ), (1, ))
    assert_size_stride(primals_5, (96, ), (1, ))
    assert_size_stride(primals_6, (96, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_8, (16, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_33, (48, ), (1, ))
    assert_size_stride(primals_34, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_36, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_39, (48, ), (1, ))
    assert_size_stride(primals_40, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_44, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_50, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (1000, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_57, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((64, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_10, buf2, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_10
        buf3 = empty_strided_cuda((64, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_16, buf3, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_16
        buf0 = empty_strided_cuda((96, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_1, buf0, 288, 49, grid=grid(288, 49), stream=stream0)
        del primals_1
        buf4 = empty_strided_cuda((128, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_22, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_22
        buf5 = empty_strided_cuda((128, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_28, buf5, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_28
        buf6 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_34, buf6, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_34
        buf7 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_40, buf7, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_40
        buf8 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_46, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_46
        buf9 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_52, buf9, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_52
        buf1 = empty_strided_cuda((4, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_3, buf1, 12, 65536, grid=grid(12, 65536), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 96, 128, 128), (1572864, 1, 12288, 96))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((4, 96, 128, 128), (1572864, 1, 12288, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_6.run(buf11, primals_2, primals_4, primals_5, primals_6, primals_7, buf12, 6291456, grid=grid(6291456), stream=stream0)
        del primals_2
        buf13 = empty_strided_cuda((4, 96, 63, 63), (381024, 1, 6048, 96), torch.float32)
        buf14 = empty_strided_cuda((4, 96, 63, 63), (381024, 1, 6048, 96), torch.int8)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_7.run(buf12, buf13, buf14, 1524096, grid=grid(1524096), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf13, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 16, 63, 63), (63504, 1, 1008, 16))
        buf87 = empty_strided_cuda((4, 16, 63, 63), (63504, 1, 1008, 16), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_8.run(buf15, primals_9, buf87, 254016, grid=grid(254016), stream=stream0)
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, x], Original ATen: [aten.convolution, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_9.run(buf16, primals_9, 254016, grid=grid(254016), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 64, 63, 63), (254016, 1, 4032, 64))
        buf86 = empty_strided_cuda((4, 64, 63, 63), (254016, 1, 4032, 64), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_10.run(buf17, primals_11, buf86, 1016064, grid=grid(1016064), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf16, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 64, 63, 63), (254016, 1, 4032, 64))
        buf85 = empty_strided_cuda((4, 64, 63, 63), (254016, 1, 4032, 64), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_10.run(buf18, primals_13, buf85, 1016064, grid=grid(1016064), stream=stream0)
        buf19 = empty_strided_cuda((4, 128, 63, 63), (508032, 1, 8064, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf17, primals_11, buf18, primals_13, buf19, 2032128, grid=grid(2032128), stream=stream0)
        del buf17
        del buf18
        del primals_11
        del primals_13
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 16, 63, 63), (63504, 1, 1008, 16))
        buf21 = empty_strided_cuda((4, 16, 63, 63), (63504, 1, 1008, 16), torch.float32)
        buf84 = empty_strided_cuda((4, 16, 63, 63), (63504, 1, 1008, 16), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_4, x_1], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_12.run(buf20, primals_15, buf21, buf84, 254016, grid=grid(254016), stream=stream0)
        del buf20
        del primals_15
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 63, 63), (254016, 1, 4032, 64))
        buf83 = empty_strided_cuda((4, 64, 63, 63), (254016, 1, 4032, 64), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_10.run(buf22, primals_17, buf83, 1016064, grid=grid(1016064), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf21, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 64, 63, 63), (254016, 1, 4032, 64))
        buf82 = empty_strided_cuda((4, 64, 63, 63), (254016, 1, 4032, 64), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_10.run(buf23, primals_19, buf82, 1016064, grid=grid(1016064), stream=stream0)
        buf24 = empty_strided_cuda((4, 128, 63, 63), (508032, 1, 8064, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf22, primals_17, buf23, primals_19, buf24, 2032128, grid=grid(2032128), stream=stream0)
        del buf22
        del buf23
        del primals_17
        del primals_19
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 32, 63, 63), (127008, 1, 2016, 32))
        buf26 = empty_strided_cuda((4, 32, 63, 63), (127008, 1, 2016, 32), torch.float32)
        buf81 = empty_strided_cuda((4, 32, 63, 63), (127008, 1, 2016, 32), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_7, x_2], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_13.run(buf25, primals_21, buf26, buf81, 508032, grid=grid(508032), stream=stream0)
        del buf25
        del primals_21
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 128, 63, 63), (508032, 1, 8064, 128))
        buf80 = empty_strided_cuda((4, 128, 63, 63), (508032, 1, 8064, 128), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_14.run(buf27, primals_23, buf80, 2032128, grid=grid(2032128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf26, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 63, 63), (508032, 1, 8064, 128))
        buf79 = empty_strided_cuda((4, 128, 63, 63), (508032, 1, 8064, 128), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_14.run(buf28, primals_25, buf79, 2032128, grid=grid(2032128), stream=stream0)
        buf29 = empty_strided_cuda((4, 256, 63, 63), (1016064, 1, 16128, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf27, primals_23, buf28, primals_25, buf29, 4064256, grid=grid(4064256), stream=stream0)
        del buf27
        del buf28
        del primals_23
        del primals_25
        buf30 = empty_strided_cuda((4, 256, 31, 31), (246016, 1, 7936, 256), torch.float32)
        buf31 = empty_strided_cuda((4, 256, 31, 31), (246016, 1, 7936, 256), torch.int8)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_16.run(buf29, buf30, buf31, 984064, grid=grid(984064), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf30, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 32, 31, 31), (30752, 1, 992, 32))
        buf33 = empty_strided_cuda((4, 32, 31, 31), (30752, 1, 992, 32), torch.float32)
        buf78 = empty_strided_cuda((4, 32, 31, 31), (30752, 1, 992, 32), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_10, x_3], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_17.run(buf32, primals_27, buf33, buf78, 123008, grid=grid(123008), stream=stream0)
        del buf32
        del primals_27
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 128, 31, 31), (123008, 1, 3968, 128))
        buf77 = empty_strided_cuda((4, 128, 31, 31), (123008, 1, 3968, 128), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_18.run(buf34, primals_29, buf77, 492032, grid=grid(492032), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf33, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 128, 31, 31), (123008, 1, 3968, 128))
        buf76 = empty_strided_cuda((4, 128, 31, 31), (123008, 1, 3968, 128), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_18.run(buf35, primals_31, buf76, 492032, grid=grid(492032), stream=stream0)
        buf36 = empty_strided_cuda((4, 256, 31, 31), (246016, 1, 7936, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf34, primals_29, buf35, primals_31, buf36, 984064, grid=grid(984064), stream=stream0)
        del buf34
        del buf35
        del primals_29
        del primals_31
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 48, 31, 31), (46128, 1, 1488, 48))
        buf38 = empty_strided_cuda((4, 48, 31, 31), (46128, 1, 1488, 48), torch.float32)
        buf75 = empty_strided_cuda((4, 48, 31, 31), (46128, 1, 1488, 48), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_13, x_4], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_20.run(buf37, primals_33, buf38, buf75, 184512, grid=grid(184512), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 192, 31, 31), (184512, 1, 5952, 192))
        buf74 = empty_strided_cuda((4, 192, 31, 31), (184512, 1, 5952, 192), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_21.run(buf39, primals_35, buf74, 738048, grid=grid(738048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf38, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 192, 31, 31), (184512, 1, 5952, 192))
        buf73 = empty_strided_cuda((4, 192, 31, 31), (184512, 1, 5952, 192), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_21.run(buf40, primals_37, buf73, 738048, grid=grid(738048), stream=stream0)
        buf41 = empty_strided_cuda((4, 384, 31, 31), (369024, 1, 11904, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_22.run(buf39, primals_35, buf40, primals_37, buf41, 1476096, grid=grid(1476096), stream=stream0)
        del buf39
        del buf40
        del primals_35
        del primals_37
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 48, 31, 31), (46128, 1, 1488, 48))
        buf43 = buf37; del buf37  # reuse
        buf72 = empty_strided_cuda((4, 48, 31, 31), (46128, 1, 1488, 48), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_16, x_5], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_20.run(buf42, primals_39, buf43, buf72, 184512, grid=grid(184512), stream=stream0)
        del buf42
        del primals_39
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 192, 31, 31), (184512, 1, 5952, 192))
        buf71 = empty_strided_cuda((4, 192, 31, 31), (184512, 1, 5952, 192), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_21.run(buf44, primals_41, buf71, 738048, grid=grid(738048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf43, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 192, 31, 31), (184512, 1, 5952, 192))
        buf70 = empty_strided_cuda((4, 192, 31, 31), (184512, 1, 5952, 192), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_21.run(buf45, primals_43, buf70, 738048, grid=grid(738048), stream=stream0)
        buf46 = empty_strided_cuda((4, 384, 31, 31), (369024, 1, 11904, 384), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_22.run(buf44, primals_41, buf45, primals_43, buf46, 1476096, grid=grid(1476096), stream=stream0)
        del buf44
        del buf45
        del primals_41
        del primals_43
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 64, 31, 31), (61504, 1, 1984, 64))
        buf48 = empty_strided_cuda((4, 64, 31, 31), (61504, 1, 1984, 64), torch.float32)
        buf69 = empty_strided_cuda((4, 64, 31, 31), (61504, 1, 1984, 64), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_19, x_6], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_23.run(buf47, primals_45, buf48, buf69, 246016, grid=grid(246016), stream=stream0)
        del buf47
        del primals_45
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 256, 31, 31), (246016, 1, 7936, 256))
        buf68 = empty_strided_cuda((4, 256, 31, 31), (246016, 1, 7936, 256), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_24.run(buf49, primals_47, buf68, 984064, grid=grid(984064), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf48, primals_48, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 256, 31, 31), (246016, 1, 7936, 256))
        buf67 = empty_strided_cuda((4, 256, 31, 31), (246016, 1, 7936, 256), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_24.run(buf50, primals_49, buf67, 984064, grid=grid(984064), stream=stream0)
        buf51 = empty_strided_cuda((4, 512, 31, 31), (492032, 1, 15872, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf49, primals_47, buf50, primals_49, buf51, 1968128, grid=grid(1968128), stream=stream0)
        del buf49
        del buf50
        del primals_47
        del primals_49
        buf52 = empty_strided_cuda((4, 512, 15, 15), (115200, 1, 7680, 512), torch.float32)
        buf53 = empty_strided_cuda((4, 512, 15, 15), (115200, 1, 7680, 512), torch.int8)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_26.run(buf51, buf52, buf53, 460800, grid=grid(460800), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf52, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 64, 15, 15), (14400, 1, 960, 64))
        buf55 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.float32)
        buf66 = empty_strided_cuda((4, 64, 15, 15), (14400, 1, 960, 64), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_22, x_7], Original ATen: [aten.convolution, aten.hardtanh, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_27.run(buf54, primals_51, buf55, buf66, 57600, grid=grid(57600), stream=stream0)
        del buf54
        del primals_51
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 15, 15), (57600, 1, 3840, 256))
        buf65 = empty_strided_cuda((4, 256, 15, 15), (57600, 1, 3840, 256), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_28.run(buf56, primals_53, buf65, 230400, grid=grid(230400), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf55, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 256, 15, 15), (57600, 1, 3840, 256))
        buf64 = empty_strided_cuda((4, 256, 15, 15), (57600, 1, 3840, 256), torch.bool)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_backward_28.run(buf57, primals_55, buf64, 230400, grid=grid(230400), stream=stream0)
        buf58 = empty_strided_cuda((4, 512, 15, 15), (115200, 1, 7680, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_29.run(buf56, primals_53, buf57, primals_55, buf58, 460800, grid=grid(460800), stream=stream0)
        del buf56
        del buf57
        del primals_53
        del primals_55
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 1000, 15, 15), (225000, 1, 15000, 1000))
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_30.run(buf60, primals_57, 900000, grid=grid(900000), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.avg_pool2d]
        buf61 = torch.ops.aten.avg_pool2d.default(buf60, [13, 13], [1, 1], [0, 0], False, True, None)
        buf62 = buf61
        del buf61
        buf63 = empty_strided_cuda((1000, 36), (36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [view], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_31.run(buf62, buf63, 36000, grid=grid(36000), stream=stream0)
        del buf62
    return (buf63, buf0, buf1, primals_4, primals_5, primals_6, primals_7, primals_8, buf2, primals_12, primals_14, buf3, primals_18, primals_20, buf4, primals_24, primals_26, buf5, primals_30, primals_32, buf6, primals_36, primals_38, buf7, primals_42, primals_44, buf8, primals_48, primals_50, buf9, primals_54, primals_56, buf11, buf12, buf13, buf14, buf16, buf19, buf21, buf24, buf26, buf29, buf30, buf31, buf33, buf36, buf38, buf41, buf43, buf46, buf48, buf51, buf52, buf53, buf55, buf58, buf60, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((96, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
