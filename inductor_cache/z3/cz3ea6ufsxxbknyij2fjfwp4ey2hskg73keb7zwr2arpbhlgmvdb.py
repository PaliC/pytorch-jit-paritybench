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


# kernel path: inductor_cache/iu/ciuihvlpugyfmbhh6eh7ftplecvjcqnmkj3ovatoemlxtxelig3y.py
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
    size_hints={'y': 512, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 75*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/f3/cf3yvvrx2dp4pn5dwcyzm6qhg7y76yekqxcww4ry23bgzk3jew7k.py
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
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/yx/cyxcpgov7jyoqlkfr6msld55v3yrsbm6lxtvvt7qdp7le7cquxyy.py
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
    size_hints={'y': 16384, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 3200*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxgirq5n3yawehg3o77ht3w4hvnbie4umck2ietblwzn54j5aqla.py
# Topologically Sorted Source Nodes: [var, pow_1, sub], Original ATen: [aten.clamp, aten.pow, aten.sub]
# Source node to ATen node mapping:
#   pow_1 => pow_1
#   sub => sub
#   var => clamp_min
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_3, 3.814697265625e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_1, 1.4551915228366852e-11), kwargs = {})
triton_poi_fused_clamp_pow_sub_3 = async_compile.triton('triton_poi_fused_clamp_pow_sub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_pow_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_pow_sub_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 3.814697265625e-06
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tmp2 * tmp2
    tmp4 = 1.4551915228366852e-11
    tmp5 = tmp3 - tmp4
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/lt/cltywsnjho7fc4nbgwtwyhxyslcfoxtlkoz2raz2wvyf5diau574.py
# Topologically Sorted Source Nodes: [pow_3], Original ATen: [aten.pow]
# Source node to ATen node mapping:
#   pow_3 => pow_3
# Graph fragment:
#   %pow_3 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution, 2), kwargs = {})
triton_poi_fused_pow_4 = async_compile.triton('triton_poi_fused_pow_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pow_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pow_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0 * tmp0
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chwxerv6hopydzggoekg3cecmy7e7cmltcbn5wl5suhh6spisxai.py
# Topologically Sorted Source Nodes: [var_1, pow_2, beta, norm_pool, norm_pool_1, norm_pool_2], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   beta => sub_1
#   norm_pool => convolution_1
#   norm_pool_1 => sqrt
#   norm_pool_2 => div
#   pow_2 => pow_2
#   var_1 => clamp_min_1
# Graph fragment:
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_4, 0.0010000072759311445), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_1, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_2, 1.4551915228366852e-11), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%pow_3, %view, %sub_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%convolution_1,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, %sqrt), kwargs = {})
triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_5 = async_compile.triton('triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp2 = 0.0010000072759311445
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp3 * tmp3
    tmp5 = 1.4551915228366852e-11
    tmp6 = tmp4 - tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = libdevice.sqrt(tmp7)
    tmp10 = tmp8 / tmp9
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/rs/crs7kt4dp4wesa3lnzbxmrrsaefyor5zrgwwajneme523yle37x6.py
# Topologically Sorted Source Nodes: [pow_6], Original ATen: [aten.pow]
# Source node to ATen node mapping:
#   pow_6 => pow_6
# Graph fragment:
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_2, 2), kwargs = {})
triton_poi_fused_pow_6 = async_compile.triton('triton_poi_fused_pow_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pow_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pow_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0 * tmp0
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckaut76nzhsy7hnxwsiy6qa577dlrrblw4bffdjuh7jle2wbrjlt.py
# Topologically Sorted Source Nodes: [var_3, pow_5, beta_1, norm_pool_3, norm_pool_4, norm_pool_5], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   beta_1 => sub_3
#   norm_pool_3 => convolution_3
#   norm_pool_4 => sqrt_1
#   norm_pool_5 => div_1
#   pow_5 => pow_5
#   var_3 => clamp_min_3
# Graph fragment:
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_7, 0.0010000072759311445), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_3, 2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_5, 1.4551915228366852e-11), kwargs = {})
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%pow_6, %view_1, %sub_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%convolution_3,), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_2, %sqrt_1), kwargs = {})
triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_7 = async_compile.triton('triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp2 = 0.0010000072759311445
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp3 * tmp3
    tmp5 = 1.4551915228366852e-11
    tmp6 = tmp4 - tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = libdevice.sqrt(tmp7)
    tmp10 = tmp8 / tmp9
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/hk/chk53w3gazcgffy56ozj6477ur7npvfphnnttsqdgb2u2rp3vex5.py
# Topologically Sorted Source Nodes: [pow_9], Original ATen: [aten.pow]
# Source node to ATen node mapping:
#   pow_9 => pow_9
# Graph fragment:
#   %pow_9 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_4, 2), kwargs = {})
triton_poi_fused_pow_8 = async_compile.triton('triton_poi_fused_pow_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pow_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pow_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0 * tmp0
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxv3vzbetflswezith4td7f2wwvpdw6ccmuryn4w7h7k2pqs43zx.py
# Topologically Sorted Source Nodes: [var_5, pow_8, beta_2, norm_pool_6, norm_pool_7, norm_pool_8], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   beta_2 => sub_5
#   norm_pool_6 => convolution_5
#   norm_pool_7 => sqrt_2
#   norm_pool_8 => div_2
#   pow_8 => pow_8
#   var_5 => clamp_min_5
# Graph fragment:
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_10, 0.0010000072759311445), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_5, 2), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_8, 1.4551915228366852e-11), kwargs = {})
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%pow_9, %view_2, %sub_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%convolution_5,), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_4, %sqrt_2), kwargs = {})
triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_9 = async_compile.triton('triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp2 = 0.0010000072759311445
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp3 * tmp3
    tmp5 = 1.4551915228366852e-11
    tmp6 = tmp4 - tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = libdevice.sqrt(tmp7)
    tmp10 = tmp8 / tmp9
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/55/c552pgat6ozvc4wojgs4hlorpsycnj6efoymktkb3q2tuy5o4p2m.py
# Topologically Sorted Source Nodes: [var_7, pow_11, beta_3, norm_pool_9, norm_pool_10, norm_pool_11], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.mul]
# Source node to ATen node mapping:
#   beta_3 => sub_7
#   norm_pool_10 => sqrt_3
#   norm_pool_11 => mul_8
#   norm_pool_9 => convolution_8
#   pow_11 => pow_11
#   var_7 => clamp_min_7
# Graph fragment:
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_14, 0.0010000072759311445), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_7, 2), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_11, 1.4551915228366852e-11), kwargs = {})
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%pow_12, %view_3, %sub_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%convolution_8,), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_7, %sqrt_3), kwargs = {})
triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_10 = async_compile.triton('triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_10(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp2 = 0.0010000072759311445
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp3 * tmp3
    tmp5 = 1.4551915228366852e-11
    tmp6 = tmp4 - tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = libdevice.sqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/4h/c4hubsepw745t5tb3frbfudk35vez7kzkp5ofujy7izohk5iydfz.py
# Topologically Sorted Source Nodes: [var_9, pow_14, beta_4, norm_pool_12, norm_pool_13, norm_pool_14], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.mul]
# Source node to ATen node mapping:
#   beta_4 => sub_9
#   norm_pool_12 => convolution_10
#   norm_pool_13 => sqrt_4
#   norm_pool_14 => mul_11
#   pow_14 => pow_14
#   var_9 => clamp_min_9
# Graph fragment:
#   %clamp_min_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_17, 0.0010000072759311445), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_9, 2), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_14, 1.4551915228366852e-11), kwargs = {})
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%pow_15, %view_4, %sub_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%convolution_10,), kwargs = {})
#   %mul_11 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_9, %sqrt_4), kwargs = {})
triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_11 = async_compile.triton('triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_11(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp2 = 0.0010000072759311445
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp3 * tmp3
    tmp5 = 1.4551915228366852e-11
    tmp6 = tmp4 - tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = libdevice.sqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/7v/c7vs2byydwabg37sb2c756xtkdqpbht2cwabgjw6qebafs7xm45l.py
# Topologically Sorted Source Nodes: [var_11, pow_17, beta_5, norm_pool_15, norm_pool_16, norm_pool_17], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.mul]
# Source node to ATen node mapping:
#   beta_5 => sub_11
#   norm_pool_15 => convolution_12
#   norm_pool_16 => sqrt_5
#   norm_pool_17 => mul_14
#   pow_17 => pow_17
#   var_11 => clamp_min_11
# Graph fragment:
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_20, 0.0010000072759311445), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_min_11, 2), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_17, 1.4551915228366852e-11), kwargs = {})
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%pow_18, %view_5, %sub_11, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%convolution_12,), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_11, %sqrt_5), kwargs = {})
triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_12 = async_compile.triton('triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_12(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp2 = 0.0010000072759311445
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp3 * tmp3
    tmp5 = 1.4551915228366852e-11
    tmp6 = tmp4 - tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = libdevice.sqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crb2mx2lpk465xfw37okxmnrge5kw75t4fitwav3f3e5vhb5vshm.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   out => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_13,), kwargs = {})
triton_poi_fused_sigmoid_13 = async_compile.triton('triton_poi_fused_sigmoid_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sigmoid_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3*x2 + 12288*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp1, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
    args.clear()
    assert_size_stride(primals_1, (128, 3, 5, 5), (75, 25, 5, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (128, 128), (128, 1))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_6, (128, 128), (128, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_9, (128, 128), (128, 1))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_12, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_13, (128, 128), (128, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_16, (128, 128), (128, 1))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_19, (128, 128), (128, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, 3, 5, 5), (75, 25, 5, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 3, 5, 5), (75, 1, 15, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 384, 25, grid=grid(384, 25), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_5, buf2, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_5
        buf3 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_8, buf3, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_8
        buf4 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_11, buf4, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_11
        buf5 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_12, buf5, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_12
        buf6 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_15, buf6, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_15
        buf7 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_18, buf7, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_18
        buf8 = empty_strided_cuda((128, 3, 5, 5), (75, 1, 15, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_21, buf8, 384, 25, grid=grid(384, 25), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf10 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var, pow_1, sub], Original ATen: [aten.clamp, aten.pow, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_pow_sub_3.run(primals_3, buf10, 16384, grid=grid(16384), stream=stream0)
        buf11 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [pow_3], Original ATen: [aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_4.run(buf9, buf11, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [var_1, pow_2, beta, norm_pool], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, reinterpret_tensor(buf10, (128, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf13 = buf12; del buf12  # reuse
        buf14 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [var_1, pow_2, beta, norm_pool, norm_pool_1, norm_pool_2], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_5.run(buf13, primals_4, buf9, buf14, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, buf2, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf16 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_2, pow_4, sub_2], Original ATen: [aten.clamp, aten.pow, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_pow_sub_3.run(primals_6, buf16, 16384, grid=grid(16384), stream=stream0)
        buf17 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [pow_6], Original ATen: [aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_6.run(buf15, buf17, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [var_3, pow_5, beta_1, norm_pool_3], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, reinterpret_tensor(buf16, (128, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [var_3, pow_5, beta_1, norm_pool_3, norm_pool_4, norm_pool_5], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_7.run(buf19, primals_7, buf15, buf20, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, buf3, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf22 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_4, pow_7, sub_4], Original ATen: [aten.clamp, aten.pow, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_pow_sub_3.run(primals_9, buf22, 16384, grid=grid(16384), stream=stream0)
        buf23 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [pow_9], Original ATen: [aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_8.run(buf21, buf23, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [var_5, pow_8, beta_2, norm_pool_6], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, reinterpret_tensor(buf22, (128, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [var_5, pow_8, beta_2, norm_pool_6, norm_pool_7, norm_pool_8], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_convolution_div_pow_sqrt_sub_9.run(buf25, primals_10, buf21, buf26, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, buf4, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 128, 4, 4), (2048, 1, 512, 128))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, buf5, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf29 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_6, pow_10, sub_6], Original ATen: [aten.clamp, aten.pow, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_pow_sub_3.run(primals_13, buf29, 16384, grid=grid(16384), stream=stream0)
        buf30 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [pow_12], Original ATen: [aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_8.run(buf28, buf30, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [var_7, pow_11, beta_3, norm_pool_9], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution]
        buf31 = extern_kernels.convolution(buf30, reinterpret_tensor(buf29, (128, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf32 = buf31; del buf31  # reuse
        buf33 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [var_7, pow_11, beta_3, norm_pool_9, norm_pool_10, norm_pool_11], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_10.run(buf32, primals_14, buf28, buf33, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, buf6, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf34, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf35 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_8, pow_13, sub_8], Original ATen: [aten.clamp, aten.pow, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_pow_sub_3.run(primals_16, buf35, 16384, grid=grid(16384), stream=stream0)
        buf36 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [pow_15], Original ATen: [aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_6.run(buf34, buf36, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [var_9, pow_14, beta_4, norm_pool_12], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, reinterpret_tensor(buf35, (128, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf38 = buf37; del buf37  # reuse
        buf39 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [var_9, pow_14, beta_4, norm_pool_12, norm_pool_13, norm_pool_14], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_11.run(buf38, primals_17, buf34, buf39, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, buf7, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf40, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf41 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_10, pow_16, sub_10], Original ATen: [aten.clamp, aten.pow, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_pow_sub_3.run(primals_19, buf41, 16384, grid=grid(16384), stream=stream0)
        buf42 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [pow_18], Original ATen: [aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_4.run(buf40, buf42, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [var_11, pow_17, beta_5, norm_pool_15], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution]
        buf43 = extern_kernels.convolution(buf42, reinterpret_tensor(buf41, (128, 128, 1, 1), (128, 1, 0, 0), 0), stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf44 = buf43; del buf43  # reuse
        buf45 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [var_11, pow_17, beta_5, norm_pool_15, norm_pool_16, norm_pool_17], Original ATen: [aten.clamp, aten.pow, aten.sub, aten.convolution, aten.sqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_convolution_mul_pow_sqrt_sub_12.run(buf44, primals_20, buf40, buf45, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, buf8, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=True, output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf46, (4, 3, 64, 64), (12288, 1, 192, 3))
        buf47 = empty_strided_cuda((4, 3, 64, 64), (12288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_13.run(buf46, buf47, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del buf46
    return (buf47, buf0, buf1, primals_3, primals_4, buf2, primals_6, primals_7, buf3, primals_9, primals_10, buf4, buf5, primals_13, primals_14, buf6, primals_16, primals_17, buf7, primals_19, primals_20, buf8, buf9, reinterpret_tensor(buf10, (128, 128, 1, 1), (128, 1, 1, 1), 0), buf11, buf13, buf14, buf15, reinterpret_tensor(buf16, (128, 128, 1, 1), (128, 1, 1, 1), 0), buf17, buf19, buf20, buf21, reinterpret_tensor(buf22, (128, 128, 1, 1), (128, 1, 1, 1), 0), buf23, buf25, buf26, buf27, buf28, reinterpret_tensor(buf29, (128, 128, 1, 1), (128, 1, 1, 1), 0), buf30, buf32, buf33, buf34, reinterpret_tensor(buf35, (128, 128, 1, 1), (128, 1, 1, 1), 0), buf36, buf38, buf39, buf40, reinterpret_tensor(buf41, (128, 128, 1, 1), (128, 1, 1, 1), 0), buf42, buf44, buf45, buf47, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 3, 5, 5), (75, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, 3, 5, 5), (75, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
