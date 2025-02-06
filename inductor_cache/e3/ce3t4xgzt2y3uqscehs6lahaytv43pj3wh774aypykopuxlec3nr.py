# AOT ID: ['5_forward']
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


# kernel path: inductor_cache/fp/cfpsqgpae3c6ansl5en6hbjxaxnizfypqzdxlu5dep4wt436coxu.py
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
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 48*y1), tmp0, xmask & ymask)
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


# kernel path: inductor_cache/li/clivct7kih3mulcdtqogdghmlqndf4p5yzeg7k3rts3tgbhzbtiz.py
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
    size_hints={'y': 32768, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 96*x2 + 384*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdx6onlficn6ytxdx6h6kzts2q7bopsc2ahhnl2j2pqws72avo2q.py
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
    size_hints={'y': 131072, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 768*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/of/coft6zimvsytm3izfap6pyxeaz43ghps2wjr5i2qltgg74sbsyav.py
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
    size_hints={'y': 524288, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 1536*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/rr/crrzwwb64e3al7r3mg2qszdo6usuq5oppr4blgn5egzl4f2dxmmo.py
# Topologically Sorted Source Nodes: [input_1, u, sub, pow_1, s, add, sqrt, x, mul, x_1], Original ATen: [aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   input_1 => convolution
#   mul => mul
#   pow_1 => pow_1
#   s => mean_1
#   sqrt => sqrt
#   sub => sub
#   u => mean
#   x => div
#   x_1 => add_1
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution, [1], True), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %mean), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %sqrt), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %div), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze_3), kwargs = {})
triton_per_fused_add_convolution_div_mean_mul_pow_sqrt_sub_5 = async_compile.triton('triton_per_fused_add_convolution_div_mean_mul_pow_sqrt_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_mean_mul_pow_sqrt_sub_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_div_mean_mul_pow_sqrt_sub_5(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 96*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 96.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp14 / tmp7
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp20 = tmp9 / tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(in_out_ptr0 + (r1 + 96*x0), tmp2, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp8, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (r1 + 96*x0), tmp23, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rr/crr3m2ltnnptooz6qtc355d6ttmyjpivruavreahqq3qgx3a73l3.py
# Topologically Sorted Source Nodes: [x_2, x_4], Original ATen: [aten.convolution, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_2 => convolution_1
#   x_4 => add_2, add_3, clone, mul_1, mul_2, rsqrt, sub_2, var_mean
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1, %primals_6, %primals_7, [1, 1], [3, 3], [1, 1], False, [0, 0], 96), kwargs = {})
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_8), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_9), kwargs = {})
triton_red_fused_convolution_native_layer_norm_6 = async_compile.triton('triton_red_fused_convolution_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x2 = (xindex % 16)
    x3 = ((xindex // 16) % 16)
    x4 = xindex // 256
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + 96*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r1 + 96*x0), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3 + 16*x2 + 256*x4), tmp4, xmask)
    tmp7 = 96.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr2 + (x3 + 16*x2 + 256*x4), tmp11, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + 96*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 - tmp4
        tmp14 = tmp13 * tmp11
        tmp16 = tmp14 * tmp15
        tmp18 = tmp16 + tmp17
        tl.store(out_ptr3 + (r1 + 96*x0), tmp18, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y6/cy6wvanl3ibpl2l4sfk5x3eynvckwt5s4u3ifsmj2z2wajs7rilj.py
# Topologically Sorted Source Nodes: [mul_1, wrapped_sqrt, pow_2, mul_2, add_2, mul_3, tanh, add_3, x_6], Original ATen: [aten.mul, aten.sqrt, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_4
#   add_3 => add_5
#   mul_1 => mul_3
#   mul_2 => mul_4
#   mul_3 => mul_5
#   pow_2 => pow_2
#   tanh => tanh
#   wrapped_sqrt => full_default
#   x_6 => mul_6
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.5), kwargs = {})
#   %full_default : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0.7978845608028654), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_1, 3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, 0.044715), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %add_4), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_5,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %add_5), kwargs = {})
triton_poi_fused_add_mul_pow_sqrt_tanh_7 = async_compile.triton('triton_poi_fused_add_mul_pow_sqrt_tanh_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_sqrt_tanh_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_sqrt_tanh_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = 0.044715
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = 0.7978845608028654
    tmp9 = tmp8 * tmp7
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp2 * tmp12
    tl.store(out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/qo/cqo2ymhtcva52g2yei6daq7sqvpk2hvxwu5c6yml7bedpu4xav2i.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_10 => add_6
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %permute_3), kwargs = {})
triton_poi_fused_add_8 = async_compile.triton('triton_poi_fused_add_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/mo/cmo7ewbf3ilidoxct7ojptiysuxbkee2ym6mwqyp5tkdo74l6j7k.py
# Topologically Sorted Source Nodes: [wrapped_sqrt, mul_6, pow_3, mul_7, add_5, mul_8, tanh_1, add_6, x_15], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_5 => add_9
#   add_6 => add_10
#   mul_6 => mul_10
#   mul_7 => mul_11
#   mul_8 => mul_12
#   pow_3 => pow_3
#   tanh_1 => tanh_1
#   wrapped_sqrt => full_default
#   x_15 => mul_13
# Graph fragment:
#   %full_default : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0.7978845608028654), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, 0.5), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_5, 3), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_3, 0.044715), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %mul_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %add_9), kwargs = {})
#   %tanh_1 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_12,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_1, 1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %add_10), kwargs = {})
triton_poi_fused_add_mul_pow_sqrt_tanh_9 = async_compile.triton('triton_poi_fused_add_mul_pow_sqrt_tanh_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_sqrt_tanh_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_sqrt_tanh_9(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 6144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 6144*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp6 * tmp5
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (y0 + 16*x2 + 98304*y1), tmp8, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 6144*y3), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qk/cqkcoh5gyqyjbmhap2bhkfgmknetci5oj3g7nf4pskxyyn7w3daj.py
# Topologically Sorted Source Nodes: [x_28, u_1, sub_2, pow_5, s_1, add_11, sqrt_1, x_29, mul_16, x_30], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_11 => add_17
#   mul_16 => mul_22
#   pow_5 => pow_5
#   s_1 => mean_3
#   sqrt_1 => sqrt_4
#   sub_2 => sub_5
#   u_1 => mean_2
#   x_28 => add_16
#   x_29 => div_1
#   x_30 => add_18
# Graph fragment:
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %permute_11), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_16, [1], True), kwargs = {})
#   %sub_5 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %mean_2), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_5, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [1], True), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %sqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_17,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, %sqrt_4), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %div_1), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_7), kwargs = {})
triton_per_fused_add_div_mean_mul_pow_sqrt_sub_10 = async_compile.triton('triton_per_fused_add_div_mean_mul_pow_sqrt_sub_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_sqrt_sub_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_pow_sqrt_sub_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 96*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 96*x0), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 96.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp4 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 / tmp9
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp22 = tmp11 / tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr1 + (r1 + 96*x0), tmp11, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr2 + (r1 + 96*x0), tmp25, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5im323dezohja4t7cgqz2wbiznkazebwr6px3qtsx7bsqx4uqob.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_2 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %primals_35, %primals_36, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/5n/c5ngepos6tyhoegx4agldkkaxh3ozy7ih6x7xbladud4npqoheos.py
# Topologically Sorted Source Nodes: [x_31, x_33], Original ATen: [aten.convolution, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_31 => convolution_5
#   x_33 => add_19, add_20, clone_3, mul_23, mul_24, rsqrt_3, sub_7, var_mean_3
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_4, %primals_37, %primals_38, [1, 1], [3, 3], [1, 1], False, [0, 0], 192), kwargs = {})
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_12,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_3, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_3, %getitem_7), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_3), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %primals_39), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %primals_40), kwargs = {})
triton_red_fused_convolution_native_layer_norm_12 = async_compile.triton('triton_red_fused_convolution_native_layer_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_layer_norm_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_layer_norm_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x2 = (xindex % 8)
    x3 = ((xindex // 8) % 8)
    x4 = xindex // 64
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + 192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r1 + 192*x0), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3 + 8*x2 + 64*x4), tmp4, xmask)
    tmp7 = 192.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr2 + (x3 + 8*x2 + 64*x4), tmp11, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + 192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 - tmp4
        tmp14 = tmp13 * tmp11
        tmp16 = tmp14 * tmp15
        tmp18 = tmp16 + tmp17
        tl.store(out_ptr3 + (r1 + 192*x0), tmp18, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ck/cck4ovzhk2jcyq52fqby2nzgxm4onnktbkmmqek2wgvk4wbjaelt.py
# Topologically Sorted Source Nodes: [wrapped_sqrt, mul_17, pow_6, mul_18, add_13, mul_19, tanh_3, add_14, x_35], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_13 => add_21
#   add_14 => add_22
#   mul_17 => mul_25
#   mul_18 => mul_26
#   mul_19 => mul_27
#   pow_6 => pow_6
#   tanh_3 => tanh_3
#   wrapped_sqrt => full_default
#   x_35 => mul_28
# Graph fragment:
#   %full_default : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0.7978845608028654), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.5), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_13, 3), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_6, 0.044715), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_13, %mul_26), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %add_21), kwargs = {})
#   %tanh_3 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_27,), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_3, 1), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %add_22), kwargs = {})
triton_poi_fused_add_mul_pow_sqrt_tanh_13 = async_compile.triton('triton_poi_fused_add_mul_pow_sqrt_tanh_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_sqrt_tanh_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_sqrt_tanh_13(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 6144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 6144*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp6 * tmp5
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (y0 + 8*x2 + 49152*y1), tmp8, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 6144*y3), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/mo/cmoikkhtvhbv7u7la7snedfyimw5ypul4ktlncqb7vu3fzkmzkwd.py
# Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_39 => add_23
# Graph fragment:
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_4, %permute_15), kwargs = {})
triton_poi_fused_add_14 = async_compile.triton('triton_poi_fused_add_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/va/cvayp6s5bavubqbryraprshz53yzxvofehvouejxbfstdd5lesxp.py
# Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_57 => add_33
# Graph fragment:
#   %add_33 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %permute_23), kwargs = {})
triton_poi_fused_add_15 = async_compile.triton('triton_poi_fused_add_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 192*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + 192*y3), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (y0 + 64*x2 + 12288*y1), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ri/crix5odxfywvmh2syrbzfv2bxszqym6tztzssz4vnjoxx34hieuy.py
# Topologically Sorted Source Nodes: [u_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   u_2 => mean_4
# Graph fragment:
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_33, [1], True), kwargs = {})
triton_red_fused_mean_16 = async_compile.triton('triton_red_fused_mean_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_16(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 6144*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cobnzpxyfof5tat6zpmztmggjatsvg3fkfx5alczw6s34atd6zed.py
# Topologically Sorted Source Nodes: [u_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   u_2 => mean_4
# Graph fragment:
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_33, [1], True), kwargs = {})
triton_per_fused_mean_17 = async_compile.triton('triton_per_fused_mean_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_17(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 128*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 192.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6j/c6jujydpqqqkytigdqi7zy5r3dtdir2i62npl4c74v6twkvrjzs6.py
# Topologically Sorted Source Nodes: [sub_4, pow_9, s_2], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   pow_9 => pow_9
#   s_2 => mean_5
#   sub_4 => sub_10
# Graph fragment:
#   %sub_10 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %mean_4), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_10, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [1], True), kwargs = {})
triton_red_fused_mean_pow_sub_18 = async_compile.triton('triton_red_fused_mean_pow_sub_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_sub_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_sub_18(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 64)
    x4 = xindex // 64
    x2 = xindex // 128
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r3 + 6144*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ju/cjuwbkomysphcu54shc4s4f4p3emaxz5dv77f5blbfgwfaazxonb.py
# Topologically Sorted Source Nodes: [sub_4, pow_9, s_2, add_22, sqrt_2], Original ATen: [aten.sub, aten.pow, aten.mean, aten.add, aten.sqrt]
# Source node to ATen node mapping:
#   add_22 => add_34
#   pow_9 => pow_9
#   s_2 => mean_5
#   sqrt_2 => sqrt_8
#   sub_4 => sub_10
# Graph fragment:
#   %sub_10 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %mean_4), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_10, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [1], True), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %sqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_34,), kwargs = {})
triton_per_fused_add_mean_pow_sqrt_sub_19 = async_compile.triton('triton_per_fused_add_mean_pow_sqrt_sub_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_pow_sqrt_sub_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_pow_sqrt_sub_19(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 128*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 192.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zl/czljpydhichoxdexmxala4vbl7kni7botlf6dwbzhy44njbbgmma.py
# Topologically Sorted Source Nodes: [sub_4, x_58, mul_32, x_59], Original ATen: [aten.sub, aten.div, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_32 => mul_44
#   sub_4 => sub_10
#   x_58 => div_2
#   x_59 => add_35
# Graph fragment:
#   %sub_10 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %mean_4), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_10, %sqrt_8), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %div_2), kwargs = {})
#   %add_35 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_11), kwargs = {})
triton_poi_fused_add_div_mul_sub_20 = async_compile.triton('triton_poi_fused_add_div_mul_sub_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sub_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 192)
    x2 = xindex
    y3 = yindex
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 / tmp4
    tmp6 = tmp0 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (y0 + 192*x2 + 12288*y1), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/5k/c5k4mqqugzezrd26hxklxrjzcvage4wmygtfmstgf4dqhcyllap5.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_3 => convolution_8
# Graph fragment:
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_35, %primals_66, %primals_67, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_21 = async_compile.triton('triton_poi_fused_convolution_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/fk/cfktqll3xgqsoy5nlgmxiysluienzf2avpo3ogmt36hsx7xc5hqw.py
# Topologically Sorted Source Nodes: [x_60, x_62], Original ATen: [aten.convolution, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_60 => convolution_9
#   x_62 => add_36, add_37, clone_6, mul_45, mul_46, rsqrt_6, sub_12, var_mean_6
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_8, %primals_68, %primals_69, [1, 1], [3, 3], [1, 1], False, [0, 0], 384), kwargs = {})
#   %clone_6 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_6, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_6, %getitem_13), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_6), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %primals_70), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %primals_71), kwargs = {})
triton_red_fused_convolution_native_layer_norm_22 = async_compile.triton('triton_red_fused_convolution_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_layer_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_layer_norm_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x2 = (xindex % 4)
    x3 = ((xindex // 4) % 4)
    x4 = xindex // 16
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + 384*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r1 + 384*x0), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3 + 4*x2 + 16*x4), tmp4, xmask)
    tmp7 = 384.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr2 + (x3 + 4*x2 + 16*x4), tmp11, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + 384*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 - tmp4
        tmp14 = tmp13 * tmp11
        tmp16 = tmp14 * tmp15
        tmp18 = tmp16 + tmp17
        tl.store(out_ptr3 + (r1 + 384*x0), tmp18, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sx/csx2mqncqshc2o4vylwofycrcxlnajsjcn33uzcsfurefnvso6gs.py
# Topologically Sorted Source Nodes: [wrapped_sqrt, mul_33, pow_10, mul_34, add_24, mul_35, tanh_6, add_25, x_64], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_24 => add_38
#   add_25 => add_39
#   mul_33 => mul_47
#   mul_34 => mul_48
#   mul_35 => mul_49
#   pow_10 => pow_10
#   tanh_6 => tanh_6
#   wrapped_sqrt => full_default
#   x_64 => mul_50
# Graph fragment:
#   %full_default : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0.7978845608028654), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, 0.5), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_25, 3), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_10, 0.044715), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_25, %mul_48), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %add_38), kwargs = {})
#   %tanh_6 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_49,), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_6, 1), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %add_39), kwargs = {})
triton_poi_fused_add_mul_pow_sqrt_tanh_23 = async_compile.triton('triton_poi_fused_add_mul_pow_sqrt_tanh_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_sqrt_tanh_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_sqrt_tanh_23(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 6144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 6144*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp6 * tmp5
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (y0 + 4*x2 + 24576*y1), tmp8, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 6144*y3), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/lq/clq3uwwz64lo3kt354czh2r2xuvauyvynzwofkiow3wpl3g7teyf.py
# Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_68 => add_40
# Graph fragment:
#   %add_40 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %permute_27), kwargs = {})
triton_poi_fused_add_24 = async_compile.triton('triton_poi_fused_add_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/bb/cbbpjpnesrisdjt4f4pydf6ohk67a4xh27cmfwecxejzpkvv3hci.py
# Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_140 => add_80
# Graph fragment:
#   %add_80 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_75, %permute_59), kwargs = {})
triton_poi_fused_add_25 = async_compile.triton('triton_poi_fused_add_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 384*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + 384*y3), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (y0 + 16*x2 + 6144*y1), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ld/cldjimzburxbaemqzfnb456emyngmntcdkgj5ruknkkzfznusy4x.py
# Topologically Sorted Source Nodes: [u_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   u_3 => mean_6
# Graph fragment:
#   %mean_6 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_80, [1], True), kwargs = {})
triton_red_fused_mean_26 = async_compile.triton('triton_red_fused_mean_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_26(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 16)
    x1 = xindex // 16
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 2048*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/at/catsebq74z7iqi3qj3e2agrnjrvtwv6yunmw4l5pa5h7wfszkeoc.py
# Topologically Sorted Source Nodes: [u_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   u_3 => mean_6
# Graph fragment:
#   %mean_6 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_80, [1], True), kwargs = {})
triton_per_fused_mean_27 = async_compile.triton('triton_per_fused_mean_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_27(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 48*x1), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 384.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ay/caycyvlhf37nq5z2f4r4lg5vgtbbzfadyjpq7lftnkx6bmbhvdhj.py
# Topologically Sorted Source Nodes: [sub_6, pow_19, s_3], Original ATen: [aten.sub, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   pow_19 => pow_19
#   s_3 => mean_7
#   sub_6 => sub_21
# Graph fragment:
#   %sub_21 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_80, %mean_6), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_21, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_19, [1], True), kwargs = {})
triton_red_fused_mean_pow_sub_28 = async_compile.triton('triton_red_fused_mean_pow_sub_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_sub_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_sub_28(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 16)
    x4 = xindex // 16
    x2 = xindex // 48
    tmp1 = tl.load(in_ptr1 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r3 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/on/conwtrbhoeldgwl7cnpxym46ijcye2kndh3l3jluyrnmsshpovlx.py
# Topologically Sorted Source Nodes: [sub_6, pow_19, s_3, add_51, sqrt_3], Original ATen: [aten.sub, aten.pow, aten.mean, aten.add, aten.sqrt]
# Source node to ATen node mapping:
#   add_51 => add_81
#   pow_19 => pow_19
#   s_3 => mean_7
#   sqrt_3 => sqrt_18
#   sub_6 => sub_21
# Graph fragment:
#   %sub_21 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_80, %mean_6), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_21, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_19, [1], True), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %sqrt_18 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_81,), kwargs = {})
triton_per_fused_add_mean_pow_sqrt_sub_29 = async_compile.triton('triton_per_fused_add_mean_pow_sqrt_sub_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_pow_sqrt_sub_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_pow_sqrt_sub_29(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 48*x1), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 384.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vy/cvyithxfccjcivacam5aovzjfak26scmtvqt7lqmxqylruxdyrzk.py
# Topologically Sorted Source Nodes: [sub_6, x_141, mul_78, x_142], Original ATen: [aten.sub, aten.div, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_78 => mul_108
#   sub_6 => sub_21
#   x_141 => div_3
#   x_142 => add_82
# Graph fragment:
#   %sub_21 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_80, %mean_6), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_21, %sqrt_18), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %div_3), kwargs = {})
#   %add_82 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, %unsqueeze_15), kwargs = {})
triton_poi_fused_add_div_mul_sub_30 = async_compile.triton('triton_poi_fused_add_div_mul_sub_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sub_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 384)
    x2 = xindex
    y3 = yindex
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 / tmp4
    tmp6 = tmp0 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (y0 + 384*x2 + 6144*y1), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/lf/clfya45rpgdz4vvvnxhabshr32wrqky62ktxxiyv7h3v5t7sam3a.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_4 => convolution_18
# Graph fragment:
#   %convolution_18 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_82, %primals_151, %primals_152, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ol/coliphs4rxgj6boypzskjfdo22bgsfzzx6b7vmweokh2m4uncwi6.py
# Topologically Sorted Source Nodes: [x_143, x_145], Original ATen: [aten.convolution, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_143 => convolution_19
#   x_145 => add_83, add_84, clone_15, mul_109, mul_110, rsqrt_15, sub_23, var_mean_15
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_18, %primals_153, %primals_154, [1, 1], [3, 3], [1, 1], False, [0, 0], 768), kwargs = {})
#   %clone_15 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_60,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_15, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-06), kwargs = {})
#   %rsqrt_15 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_83,), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_15, %getitem_31), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_15), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %primals_155), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %primals_156), kwargs = {})
triton_red_fused_convolution_native_layer_norm_32 = async_compile.triton('triton_red_fused_convolution_native_layer_norm_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_layer_norm_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_layer_norm_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x2 = (xindex % 2)
    x3 = ((xindex // 2) % 2)
    x4 = xindex // 4
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r1 + 768*x0), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3 + 2*x2 + 4*x4), tmp4, xmask)
    tmp7 = 768.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr2 + (x3 + 2*x2 + 4*x4), tmp11, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 - tmp4
        tmp14 = tmp13 * tmp11
        tmp16 = tmp14 * tmp15
        tmp18 = tmp16 + tmp17
        tl.store(out_ptr3 + (r1 + 768*x0), tmp18, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4n/c4nkrwp6be2ogbx5ka2eic3cwlkfflhxjrq47e3utyyelfokvjwu.py
# Topologically Sorted Source Nodes: [wrapped_sqrt, mul_79, pow_20, mul_80, add_53, mul_81, tanh_15, add_54, x_147], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_53 => add_85
#   add_54 => add_86
#   mul_79 => mul_111
#   mul_80 => mul_112
#   mul_81 => mul_113
#   pow_20 => pow_20
#   tanh_15 => tanh_15
#   wrapped_sqrt => full_default
#   x_147 => mul_114
# Graph fragment:
#   %full_default : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0.7978845608028654), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_61, 0.5), kwargs = {})
#   %pow_20 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_61, 3), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_20, 0.044715), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_61, %mul_112), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %add_85), kwargs = {})
#   %tanh_15 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_113,), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_15, 1), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_111, %add_86), kwargs = {})
triton_poi_fused_add_mul_pow_sqrt_tanh_33 = async_compile.triton('triton_poi_fused_add_mul_pow_sqrt_tanh_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_sqrt_tanh_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_sqrt_tanh_33(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 6144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2)
    y1 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 6144*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp6 * tmp5
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (y0 + 2*x2 + 12288*y1), tmp8, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 6144*y3), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/g7/cg7o772spxsa5rwgqbrjntv6atxk56uaqycqo7cfs27bktfthw2g.py
# Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_151 => add_87
# Graph fragment:
#   %add_87 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_18, %permute_63), kwargs = {})
triton_poi_fused_add_34 = async_compile.triton('triton_poi_fused_add_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/2u/c2udsxmfbwm4iqkhqh6pgd7x5kny6p7pfq4ak6yycakcc7ay434i.py
# Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_169 => add_97
# Graph fragment:
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %permute_71), kwargs = {})
triton_poi_fused_add_35 = async_compile.triton('triton_poi_fused_add_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 768*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + 768*y3), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (y0 + 4*x2 + 3072*y1), tmp4, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179 = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (96, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (96, ), (1, ))
    assert_size_stride(primals_5, (96, ), (1, ))
    assert_size_stride(primals_6, (96, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (384, 96), (96, 1))
    assert_size_stride(primals_11, (384, ), (1, ))
    assert_size_stride(primals_12, (96, 384), (384, 1))
    assert_size_stride(primals_13, (96, ), (1, ))
    assert_size_stride(primals_14, (96, ), (1, ))
    assert_size_stride(primals_15, (96, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_16, (96, ), (1, ))
    assert_size_stride(primals_17, (96, ), (1, ))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (384, 96), (96, 1))
    assert_size_stride(primals_20, (384, ), (1, ))
    assert_size_stride(primals_21, (96, 384), (384, 1))
    assert_size_stride(primals_22, (96, ), (1, ))
    assert_size_stride(primals_23, (96, ), (1, ))
    assert_size_stride(primals_24, (96, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_25, (96, ), (1, ))
    assert_size_stride(primals_26, (96, ), (1, ))
    assert_size_stride(primals_27, (96, ), (1, ))
    assert_size_stride(primals_28, (384, 96), (96, 1))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (96, 384), (384, 1))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_32, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (192, 96, 2, 2), (384, 4, 2, 1))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (768, 192), (192, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (192, 768), (768, 1))
    assert_size_stride(primals_44, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_50, (768, 192), (192, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (192, 768), (768, 1))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_54, (192, ), (1, ))
    assert_size_stride(primals_55, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_57, (192, ), (1, ))
    assert_size_stride(primals_58, (192, ), (1, ))
    assert_size_stride(primals_59, (768, 192), (192, 1))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (192, 768), (768, 1))
    assert_size_stride(primals_62, (192, ), (1, ))
    assert_size_stride(primals_63, (192, ), (1, ))
    assert_size_stride(primals_64, (192, ), (1, ))
    assert_size_stride(primals_65, (192, ), (1, ))
    assert_size_stride(primals_66, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_72, (1536, 384), (384, 1))
    assert_size_stride(primals_73, (1536, ), (1, ))
    assert_size_stride(primals_74, (384, 1536), (1536, 1))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_80, (384, ), (1, ))
    assert_size_stride(primals_81, (1536, 384), (384, 1))
    assert_size_stride(primals_82, (1536, ), (1, ))
    assert_size_stride(primals_83, (384, 1536), (1536, 1))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (1536, 384), (384, 1))
    assert_size_stride(primals_91, (1536, ), (1, ))
    assert_size_stride(primals_92, (384, 1536), (1536, 1))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (384, ), (1, ))
    assert_size_stride(primals_98, (384, ), (1, ))
    assert_size_stride(primals_99, (1536, 384), (384, 1))
    assert_size_stride(primals_100, (1536, ), (1, ))
    assert_size_stride(primals_101, (384, 1536), (1536, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (384, ), (1, ))
    assert_size_stride(primals_104, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (1536, 384), (384, 1))
    assert_size_stride(primals_109, (1536, ), (1, ))
    assert_size_stride(primals_110, (384, 1536), (1536, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (384, ), (1, ))
    assert_size_stride(primals_116, (384, ), (1, ))
    assert_size_stride(primals_117, (1536, 384), (384, 1))
    assert_size_stride(primals_118, (1536, ), (1, ))
    assert_size_stride(primals_119, (384, 1536), (1536, 1))
    assert_size_stride(primals_120, (384, ), (1, ))
    assert_size_stride(primals_121, (384, ), (1, ))
    assert_size_stride(primals_122, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (384, ), (1, ))
    assert_size_stride(primals_126, (1536, 384), (384, 1))
    assert_size_stride(primals_127, (1536, ), (1, ))
    assert_size_stride(primals_128, (384, 1536), (1536, 1))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_132, (384, ), (1, ))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (1536, 384), (384, 1))
    assert_size_stride(primals_136, (1536, ), (1, ))
    assert_size_stride(primals_137, (384, 1536), (1536, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (384, ), (1, ))
    assert_size_stride(primals_144, (1536, 384), (384, 1))
    assert_size_stride(primals_145, (1536, ), (1, ))
    assert_size_stride(primals_146, (384, 1536), (1536, 1))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (768, 384, 2, 2), (1536, 4, 2, 1))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, ), (1, ))
    assert_size_stride(primals_157, (3072, 768), (768, 1))
    assert_size_stride(primals_158, (3072, ), (1, ))
    assert_size_stride(primals_159, (768, 3072), (3072, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, ), (1, ))
    assert_size_stride(primals_166, (3072, 768), (768, 1))
    assert_size_stride(primals_167, (3072, ), (1, ))
    assert_size_stride(primals_168, (768, 3072), (3072, 1))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (3072, 768), (768, 1))
    assert_size_stride(primals_176, (3072, ), (1, ))
    assert_size_stride(primals_177, (768, 3072), (3072, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((96, 3, 4, 4), (48, 1, 12, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 288, 16, grid=grid(288, 16), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_3, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((192, 96, 2, 2), (384, 1, 192, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_35, buf2, 18432, 4, grid=grid(18432, 4), stream=stream0)
        del primals_35
        buf3 = empty_strided_cuda((384, 192, 2, 2), (768, 1, 384, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_66, buf3, 73728, 4, grid=grid(73728, 4), stream=stream0)
        del primals_66
        buf4 = empty_strided_cuda((768, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_151, buf4, 294912, 4, grid=grid(294912, 4), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf1, buf0, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 96, 16, 16), (24576, 1, 1536, 96))
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided_cuda((4, 1, 16, 16), (256, 1024, 16, 1), torch.float32)
        buf8 = reinterpret_tensor(buf7, (4, 1, 16, 16), (256, 1, 16, 1), 0); del buf7  # reuse
        buf9 = empty_strided_cuda((4, 1, 16, 16), (256, 1024, 16, 1), torch.float32)
        buf10 = reinterpret_tensor(buf9, (4, 1, 16, 16), (256, 1, 16, 1), 0); del buf9  # reuse
        buf11 = empty_strided_cuda((4, 96, 16, 16), (24576, 1, 1536, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, u, sub, pow_1, s, add, sqrt, x, mul, x_1], Original ATen: [aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_mean_mul_pow_sqrt_sub_5.run(buf6, buf8, buf10, primals_2, primals_4, primals_5, buf11, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_2
        del primals_5
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_6, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf12, (4, 96, 16, 16), (24576, 1, 1536, 96))
        buf13 = buf12; del buf12  # reuse
        buf14 = empty_strided_cuda((4, 16, 16, 1), (256, 1, 16, 16), torch.float32)
        buf17 = empty_strided_cuda((4, 16, 16, 1), (256, 1, 16, 16), torch.float32)
        buf18 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_4], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_6.run(buf13, primals_7, primals_8, primals_9, buf14, buf17, buf18, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_7
        del primals_9
        buf19 = empty_strided_cuda((1024, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf18, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_10, (96, 384), (1, 96), 0), alpha=1, beta=1, out=buf19)
        del primals_11
        buf20 = empty_strided_cuda((4, 16, 16, 384), (98304, 6144, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, wrapped_sqrt, pow_2, mul_2, add_2, mul_3, tanh, add_3, x_6], Original ATen: [aten.mul, aten.sqrt, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_7.run(buf19, buf20, 393216, grid=grid(393216), stream=stream0)
        buf21 = empty_strided_cuda((1024, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf20, (1024, 384), (384, 1), 0), reinterpret_tensor(primals_12, (384, 96), (1, 384), 0), alpha=1, beta=1, out=buf21)
        del primals_13
        buf22 = empty_strided_cuda((4, 96, 16, 16), (24576, 1, 1536, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_8.run(buf11, primals_14, buf21, buf22, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_15, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf23, (4, 96, 16, 16), (24576, 1, 1536, 96))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 16, 16, 1), (256, 1, 16, 16), torch.float32)
        buf28 = empty_strided_cuda((4, 16, 16, 1), (256, 1, 16, 16), torch.float32)
        buf29 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_13], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_6.run(buf24, primals_16, primals_17, primals_18, buf25, buf28, buf29, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_16
        del primals_18
        buf30 = empty_strided_cuda((1024, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_20, reinterpret_tensor(buf29, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_19, (96, 384), (1, 96), 0), alpha=1, beta=1, out=buf30)
        del primals_20
        buf31 = empty_strided_cuda((4, 16, 16, 384), (98304, 1, 6144, 16), torch.float32)
        buf32 = empty_strided_cuda((4, 16, 16, 384), (98304, 6144, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_6, pow_3, mul_7, add_5, mul_8, tanh_1, add_6, x_15], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_9.run(buf30, buf31, buf32, 64, 6144, grid=grid(64, 6144), stream=stream0)
        buf33 = empty_strided_cuda((1024, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_22, reinterpret_tensor(buf32, (1024, 384), (384, 1), 0), reinterpret_tensor(primals_21, (384, 96), (1, 384), 0), alpha=1, beta=1, out=buf33)
        del primals_22
        buf34 = empty_strided_cuda((4, 96, 16, 16), (24576, 1, 1536, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_8.run(buf22, primals_23, buf33, buf34, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_24, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf35, (4, 96, 16, 16), (24576, 1, 1536, 96))
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((4, 16, 16, 1), (256, 1, 16, 16), torch.float32)
        buf40 = empty_strided_cuda((4, 16, 16, 1), (256, 1, 16, 16), torch.float32)
        buf41 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20, x_22], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_6.run(buf36, primals_25, primals_26, primals_27, buf37, buf40, buf41, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_25
        del primals_27
        buf42 = empty_strided_cuda((1024, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, reinterpret_tensor(buf41, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_28, (96, 384), (1, 96), 0), alpha=1, beta=1, out=buf42)
        del primals_29
        buf43 = empty_strided_cuda((4, 16, 16, 384), (98304, 1, 6144, 16), torch.float32)
        buf44 = empty_strided_cuda((4, 16, 16, 384), (98304, 6144, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_11, pow_4, mul_12, add_8, mul_13, tanh_2, add_9, x_24], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_9.run(buf42, buf43, buf44, 64, 6144, grid=grid(64, 6144), stream=stream0)
        buf45 = empty_strided_cuda((1024, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_31, reinterpret_tensor(buf44, (1024, 384), (384, 1), 0), reinterpret_tensor(primals_30, (384, 96), (1, 384), 0), alpha=1, beta=1, out=buf45)
        del primals_31
        buf47 = empty_strided_cuda((4, 96, 16, 16), (24576, 1, 1536, 96), torch.float32)
        buf48 = empty_strided_cuda((4, 1, 16, 16), (256, 1024, 16, 1), torch.float32)
        buf49 = reinterpret_tensor(buf48, (4, 1, 16, 16), (256, 1, 16, 1), 0); del buf48  # reuse
        buf50 = empty_strided_cuda((4, 96, 16, 16), (24576, 1, 1536, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_28, u_1, sub_2, pow_5, s_1, add_11, sqrt_1, x_29, mul_16, x_30], Original ATen: [aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_10.run(buf49, buf34, primals_32, buf45, primals_33, primals_34, buf47, buf50, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_34
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, buf2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 192, 8, 8), (12288, 1, 1536, 192))
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_11.run(buf52, primals_36, 49152, grid=grid(49152), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_37, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf53, (4, 192, 8, 8), (12288, 1, 1536, 192))
        buf54 = buf53; del buf53  # reuse
        buf55 = empty_strided_cuda((4, 8, 8, 1), (64, 1, 8, 8), torch.float32)
        buf58 = empty_strided_cuda((4, 8, 8, 1), (64, 1, 8, 8), torch.float32)
        buf59 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_31, x_33], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_12.run(buf54, primals_38, primals_39, primals_40, buf55, buf58, buf59, 256, 192, grid=grid(256), stream=stream0)
        del primals_38
        del primals_40
        buf60 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_42, reinterpret_tensor(buf59, (256, 192), (192, 1), 0), reinterpret_tensor(primals_41, (192, 768), (1, 192), 0), alpha=1, beta=1, out=buf60)
        del primals_42
        buf61 = empty_strided_cuda((4, 8, 8, 768), (49152, 1, 6144, 8), torch.float32)
        buf62 = empty_strided_cuda((4, 8, 8, 768), (49152, 6144, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_17, pow_6, mul_18, add_13, mul_19, tanh_3, add_14, x_35], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_13.run(buf60, buf61, buf62, 32, 6144, grid=grid(32, 6144), stream=stream0)
        buf63 = empty_strided_cuda((256, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_44, reinterpret_tensor(buf62, (256, 768), (768, 1), 0), reinterpret_tensor(primals_43, (768, 192), (1, 768), 0), alpha=1, beta=1, out=buf63)
        del primals_44
        buf64 = empty_strided_cuda((4, 192, 8, 8), (12288, 1, 1536, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_14.run(buf52, primals_45, buf63, buf64, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_46, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf65, (4, 192, 8, 8), (12288, 1, 1536, 192))
        buf66 = buf65; del buf65  # reuse
        buf67 = empty_strided_cuda((4, 8, 8, 1), (64, 1, 8, 8), torch.float32)
        buf70 = empty_strided_cuda((4, 8, 8, 1), (64, 1, 8, 8), torch.float32)
        buf71 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_40, x_42], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_12.run(buf66, primals_47, primals_48, primals_49, buf67, buf70, buf71, 256, 192, grid=grid(256), stream=stream0)
        del primals_47
        del primals_49
        buf72 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, reinterpret_tensor(buf71, (256, 192), (192, 1), 0), reinterpret_tensor(primals_50, (192, 768), (1, 192), 0), alpha=1, beta=1, out=buf72)
        del primals_51
        buf73 = empty_strided_cuda((4, 8, 8, 768), (49152, 1, 6144, 8), torch.float32)
        buf74 = empty_strided_cuda((4, 8, 8, 768), (49152, 6144, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_22, pow_7, mul_23, add_16, mul_24, tanh_4, add_17, x_44], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_13.run(buf72, buf73, buf74, 32, 6144, grid=grid(32, 6144), stream=stream0)
        buf75 = empty_strided_cuda((256, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_53, reinterpret_tensor(buf74, (256, 768), (768, 1), 0), reinterpret_tensor(primals_52, (768, 192), (1, 768), 0), alpha=1, beta=1, out=buf75)
        del primals_53
        buf76 = empty_strided_cuda((4, 192, 8, 8), (12288, 1, 1536, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_14.run(buf64, primals_54, buf75, buf76, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_55, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf77, (4, 192, 8, 8), (12288, 1, 1536, 192))
        buf78 = buf77; del buf77  # reuse
        buf79 = empty_strided_cuda((4, 8, 8, 1), (64, 1, 8, 8), torch.float32)
        buf82 = empty_strided_cuda((4, 8, 8, 1), (64, 1, 8, 8), torch.float32)
        buf83 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_49, x_51], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_12.run(buf78, primals_56, primals_57, primals_58, buf79, buf82, buf83, 256, 192, grid=grid(256), stream=stream0)
        del primals_56
        del primals_58
        buf84 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, reinterpret_tensor(buf83, (256, 192), (192, 1), 0), reinterpret_tensor(primals_59, (192, 768), (1, 192), 0), alpha=1, beta=1, out=buf84)
        del primals_60
        buf85 = empty_strided_cuda((4, 8, 8, 768), (49152, 1, 6144, 8), torch.float32)
        buf86 = empty_strided_cuda((4, 8, 8, 768), (49152, 6144, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_27, pow_8, mul_28, add_19, mul_29, tanh_5, add_20, x_53], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_13.run(buf84, buf85, buf86, 32, 6144, grid=grid(32, 6144), stream=stream0)
        buf87 = empty_strided_cuda((256, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_62, reinterpret_tensor(buf86, (256, 768), (768, 1), 0), reinterpret_tensor(primals_61, (768, 192), (1, 768), 0), alpha=1, beta=1, out=buf87)
        del primals_62
        buf88 = empty_strided_cuda((4, 192, 8, 8), (12288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_15.run(buf76, primals_63, buf87, buf88, 256, 192, grid=grid(256, 192), stream=stream0)
        buf89 = empty_strided_cuda((4, 1, 8, 8, 2), (128, 512, 8, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [u_2], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_16.run(buf88, buf89, 512, 96, grid=grid(512), stream=stream0)
        buf90 = empty_strided_cuda((4, 1, 8, 8), (64, 256, 8, 1), torch.float32)
        buf91 = reinterpret_tensor(buf90, (4, 1, 8, 8), (64, 1, 8, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [u_2], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_17.run(buf91, buf89, 256, 2, grid=grid(256), stream=stream0)
        buf92 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [sub_4, pow_9, s_2], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_sub_18.run(buf88, buf91, buf92, 512, 96, grid=grid(512), stream=stream0)
        buf93 = empty_strided_cuda((4, 1, 8, 8), (64, 256, 8, 1), torch.float32)
        buf94 = reinterpret_tensor(buf93, (4, 1, 8, 8), (64, 1, 8, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [sub_4, pow_9, s_2, add_22, sqrt_2], Original ATen: [aten.sub, aten.pow, aten.mean, aten.add, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_pow_sqrt_sub_19.run(buf94, buf92, 256, 2, grid=grid(256), stream=stream0)
        del buf92
        buf95 = empty_strided_cuda((4, 192, 8, 8), (12288, 1, 1536, 192), torch.float32)
        # Topologically Sorted Source Nodes: [sub_4, x_58, mul_32, x_59], Original ATen: [aten.sub, aten.div, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_20.run(primals_64, buf88, buf91, buf94, primals_65, buf95, 768, 64, grid=grid(768, 64), stream=stream0)
        del primals_65
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, buf3, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(buf97, primals_67, 24576, grid=grid(24576), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_68, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf98, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf99 = buf98; del buf98  # reuse
        buf100 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf103 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf104 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, x_62], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf99, primals_69, primals_70, primals_71, buf100, buf103, buf104, 64, 384, grid=grid(64), stream=stream0)
        del primals_69
        del primals_71
        buf105 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_73, reinterpret_tensor(buf104, (64, 384), (384, 1), 0), reinterpret_tensor(primals_72, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf105)
        del primals_73
        buf106 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf107 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_33, pow_10, mul_34, add_24, mul_35, tanh_6, add_25, x_64], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf105, buf106, buf107, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf108 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_75, reinterpret_tensor(buf107, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_74, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf108)
        del primals_75
        buf109 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf97, primals_76, buf108, buf109, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_77, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf110, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf111 = buf110; del buf110  # reuse
        buf112 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf115 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf116 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_69, x_71], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf111, primals_78, primals_79, primals_80, buf112, buf115, buf116, 64, 384, grid=grid(64), stream=stream0)
        del primals_78
        del primals_80
        buf117 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_82, reinterpret_tensor(buf116, (64, 384), (384, 1), 0), reinterpret_tensor(primals_81, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf117)
        del primals_82
        buf118 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf119 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_38, pow_11, mul_39, add_27, mul_40, tanh_7, add_28, x_73], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf117, buf118, buf119, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf120 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_84, reinterpret_tensor(buf119, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_83, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf120)
        del primals_84
        buf121 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf109, primals_85, buf120, buf121, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_86, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf122, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf123 = buf122; del buf122  # reuse
        buf124 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf127 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf128 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_78, x_80], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf123, primals_87, primals_88, primals_89, buf124, buf127, buf128, 64, 384, grid=grid(64), stream=stream0)
        del primals_87
        del primals_89
        buf129 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_91, reinterpret_tensor(buf128, (64, 384), (384, 1), 0), reinterpret_tensor(primals_90, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf129)
        del primals_91
        buf130 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf131 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_43, pow_12, mul_44, add_30, mul_45, tanh_8, add_31, x_82], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf129, buf130, buf131, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf132 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_83], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, reinterpret_tensor(buf131, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_92, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf132)
        del primals_93
        buf133 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf121, primals_94, buf132, buf133, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_95, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf134, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf135 = buf134; del buf134  # reuse
        buf136 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf139 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf140 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_87, x_89], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf135, primals_96, primals_97, primals_98, buf136, buf139, buf140, 64, 384, grid=grid(64), stream=stream0)
        del primals_96
        del primals_98
        buf141 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_100, reinterpret_tensor(buf140, (64, 384), (384, 1), 0), reinterpret_tensor(primals_99, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf141)
        del primals_100
        buf142 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf143 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_48, pow_13, mul_49, add_33, mul_50, tanh_9, add_34, x_91], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf141, buf142, buf143, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf144 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, reinterpret_tensor(buf143, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_101, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf144)
        del primals_102
        buf145 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_95], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf133, primals_103, buf144, buf145, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_104, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf146, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf147 = buf146; del buf146  # reuse
        buf148 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf151 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf152 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_96, x_98], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf147, primals_105, primals_106, primals_107, buf148, buf151, buf152, 64, 384, grid=grid(64), stream=stream0)
        del primals_105
        del primals_107
        buf153 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_109, reinterpret_tensor(buf152, (64, 384), (384, 1), 0), reinterpret_tensor(primals_108, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf153)
        del primals_109
        buf154 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf155 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_53, pow_14, mul_54, add_36, mul_55, tanh_10, add_37, x_100], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf153, buf154, buf155, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf156 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_111, reinterpret_tensor(buf155, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_110, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf156)
        del primals_111
        buf157 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf145, primals_112, buf156, buf157, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_113, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf158, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf159 = buf158; del buf158  # reuse
        buf160 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf163 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf164 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_105, x_107], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf159, primals_114, primals_115, primals_116, buf160, buf163, buf164, 64, 384, grid=grid(64), stream=stream0)
        del primals_114
        del primals_116
        buf165 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_118, reinterpret_tensor(buf164, (64, 384), (384, 1), 0), reinterpret_tensor(primals_117, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf165)
        del primals_118
        buf166 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf167 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_58, pow_15, mul_59, add_39, mul_60, tanh_11, add_40, x_109], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf165, buf166, buf167, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf168 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, reinterpret_tensor(buf167, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_119, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf168)
        del primals_120
        buf169 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_113], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf157, primals_121, buf168, buf169, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_122, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf170, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf171 = buf170; del buf170  # reuse
        buf172 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf175 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf176 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_114, x_116], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf171, primals_123, primals_124, primals_125, buf172, buf175, buf176, 64, 384, grid=grid(64), stream=stream0)
        del primals_123
        del primals_125
        buf177 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_117], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_127, reinterpret_tensor(buf176, (64, 384), (384, 1), 0), reinterpret_tensor(primals_126, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf177)
        del primals_127
        buf178 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf179 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_63, pow_16, mul_64, add_42, mul_65, tanh_12, add_43, x_118], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf177, buf178, buf179, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf180 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, reinterpret_tensor(buf179, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_128, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf180)
        del primals_129
        buf181 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf169, primals_130, buf180, buf181, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_123], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_131, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf182, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf183 = buf182; del buf182  # reuse
        buf184 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf187 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf188 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_123, x_125], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf183, primals_132, primals_133, primals_134, buf184, buf187, buf188, 64, 384, grid=grid(64), stream=stream0)
        del primals_132
        del primals_134
        buf189 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, reinterpret_tensor(buf188, (64, 384), (384, 1), 0), reinterpret_tensor(primals_135, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf189)
        del primals_136
        buf190 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf191 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_68, pow_17, mul_69, add_45, mul_70, tanh_13, add_46, x_127], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf189, buf190, buf191, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf192 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_138, reinterpret_tensor(buf191, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_137, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf192)
        del primals_138
        buf193 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_131], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_24.run(buf181, primals_139, buf192, buf193, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_140, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf194, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf195 = buf194; del buf194  # reuse
        buf196 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf199 = empty_strided_cuda((4, 4, 4, 1), (16, 1, 4, 4), torch.float32)
        buf200 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_132, x_134], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_22.run(buf195, primals_141, primals_142, primals_143, buf196, buf199, buf200, 64, 384, grid=grid(64), stream=stream0)
        del primals_141
        del primals_143
        buf201 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, reinterpret_tensor(buf200, (64, 384), (384, 1), 0), reinterpret_tensor(primals_144, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf201)
        del primals_145
        buf202 = empty_strided_cuda((4, 4, 4, 1536), (24576, 1, 6144, 4), torch.float32)
        buf203 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_73, pow_18, mul_74, add_48, mul_75, tanh_14, add_49, x_136], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_23.run(buf201, buf202, buf203, 16, 6144, grid=grid(16, 6144), stream=stream0)
        buf204 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_137], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_147, reinterpret_tensor(buf203, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_146, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf204)
        del primals_147
        buf205 = empty_strided_cuda((4, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_25.run(buf193, primals_148, buf204, buf205, 64, 384, grid=grid(64, 384), stream=stream0)
        buf206 = empty_strided_cuda((4, 1, 4, 4, 3), (48, 192, 4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [u_3], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_26.run(buf205, buf206, 192, 128, grid=grid(192), stream=stream0)
        buf207 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf208 = reinterpret_tensor(buf207, (4, 1, 4, 4), (16, 1, 4, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [u_3], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_27.run(buf208, buf206, 64, 3, grid=grid(64), stream=stream0)
        buf209 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [sub_6, pow_19, s_3], Original ATen: [aten.sub, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_sub_28.run(buf205, buf208, buf209, 192, 128, grid=grid(192), stream=stream0)
        buf210 = empty_strided_cuda((4, 1, 4, 4), (16, 64, 4, 1), torch.float32)
        buf211 = reinterpret_tensor(buf210, (4, 1, 4, 4), (16, 1, 4, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [sub_6, pow_19, s_3, add_51, sqrt_3], Original ATen: [aten.sub, aten.pow, aten.mean, aten.add, aten.sqrt]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_pow_sqrt_sub_29.run(buf211, buf209, 64, 3, grid=grid(64), stream=stream0)
        del buf209
        buf212 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [sub_6, x_141, mul_78, x_142], Original ATen: [aten.sub, aten.div, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_30.run(primals_149, buf205, buf208, buf211, primals_150, buf212, 1536, 16, grid=grid(1536, 16), stream=stream0)
        del primals_150
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, buf4, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 768, 2, 2), (3072, 1, 1536, 768))
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf214, primals_152, 12288, grid=grid(12288), stream=stream0)
        del primals_152
        # Topologically Sorted Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_153, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf215, (4, 768, 2, 2), (3072, 1, 1536, 768))
        buf216 = buf215; del buf215  # reuse
        buf217 = empty_strided_cuda((4, 2, 2, 1), (4, 1, 2, 2), torch.float32)
        buf220 = empty_strided_cuda((4, 2, 2, 1), (4, 1, 2, 2), torch.float32)
        buf221 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_143, x_145], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_32.run(buf216, primals_154, primals_155, primals_156, buf217, buf220, buf221, 16, 768, grid=grid(16), stream=stream0)
        del primals_154
        del primals_156
        buf222 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, reinterpret_tensor(buf221, (16, 768), (768, 1), 0), reinterpret_tensor(primals_157, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf222)
        del primals_158
        buf223 = empty_strided_cuda((4, 2, 2, 3072), (12288, 1, 6144, 2), torch.float32)
        buf224 = empty_strided_cuda((4, 2, 2, 3072), (12288, 6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_79, pow_20, mul_80, add_53, mul_81, tanh_15, add_54, x_147], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_33.run(buf222, buf223, buf224, 8, 6144, grid=grid(8, 6144), stream=stream0)
        buf225 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_160, reinterpret_tensor(buf224, (16, 3072), (3072, 1), 0), reinterpret_tensor(primals_159, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf225)
        del primals_160
        buf226 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_34.run(buf214, primals_161, buf225, buf226, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_162, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf227, (4, 768, 2, 2), (3072, 1, 1536, 768))
        buf228 = buf227; del buf227  # reuse
        buf229 = empty_strided_cuda((4, 2, 2, 1), (4, 1, 2, 2), torch.float32)
        buf232 = empty_strided_cuda((4, 2, 2, 1), (4, 1, 2, 2), torch.float32)
        buf233 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_152, x_154], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_32.run(buf228, primals_163, primals_164, primals_165, buf229, buf232, buf233, 16, 768, grid=grid(16), stream=stream0)
        del primals_163
        del primals_165
        buf234 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_155], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_167, reinterpret_tensor(buf233, (16, 768), (768, 1), 0), reinterpret_tensor(primals_166, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf234)
        del primals_167
        buf235 = empty_strided_cuda((4, 2, 2, 3072), (12288, 1, 6144, 2), torch.float32)
        buf236 = empty_strided_cuda((4, 2, 2, 3072), (12288, 6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_84, pow_21, mul_85, add_56, mul_86, tanh_16, add_57, x_156], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_33.run(buf234, buf235, buf236, 8, 6144, grid=grid(8, 6144), stream=stream0)
        buf237 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_157], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_169, reinterpret_tensor(buf236, (16, 3072), (3072, 1), 0), reinterpret_tensor(primals_168, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf237)
        del primals_169
        buf238 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_34.run(buf226, primals_170, buf237, buf238, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_171, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf239, (4, 768, 2, 2), (3072, 1, 1536, 768))
        buf240 = buf239; del buf239  # reuse
        buf241 = empty_strided_cuda((4, 2, 2, 1), (4, 1, 2, 2), torch.float32)
        buf244 = empty_strided_cuda((4, 2, 2, 1), (4, 1, 2, 2), torch.float32)
        buf245 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_161, x_163], Original ATen: [aten.convolution, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_layer_norm_32.run(buf240, primals_172, primals_173, primals_174, buf241, buf244, buf245, 16, 768, grid=grid(16), stream=stream0)
        del primals_172
        del primals_174
        buf246 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, reinterpret_tensor(buf245, (16, 768), (768, 1), 0), reinterpret_tensor(primals_175, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf246)
        del primals_176
        buf247 = empty_strided_cuda((4, 2, 2, 3072), (12288, 1, 6144, 2), torch.float32)
        buf248 = empty_strided_cuda((4, 2, 2, 3072), (12288, 6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [wrapped_sqrt, mul_89, pow_22, mul_90, add_59, mul_91, tanh_17, add_60, x_165], Original ATen: [aten.sqrt, aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sqrt_tanh_33.run(buf246, buf247, buf248, 8, 6144, grid=grid(8, 6144), stream=stream0)
        buf249 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_178, reinterpret_tensor(buf248, (16, 3072), (3072, 1), 0), reinterpret_tensor(primals_177, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf249)
        del primals_178
        buf250 = empty_strided_cuda((4, 768, 2, 2), (3072, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_35.run(buf238, primals_179, buf249, buf250, 16, 768, grid=grid(16, 768), stream=stream0)
    return (buf88, buf205, buf250, buf0, buf1, primals_4, primals_6, primals_8, primals_14, primals_15, primals_17, primals_23, primals_24, primals_26, primals_32, primals_33, buf2, primals_37, primals_39, primals_45, primals_46, primals_48, primals_54, primals_55, primals_57, primals_63, primals_64, buf3, primals_68, primals_70, primals_76, primals_77, primals_79, primals_85, primals_86, primals_88, primals_94, primals_95, primals_97, primals_103, primals_104, primals_106, primals_112, primals_113, primals_115, primals_121, primals_122, primals_124, primals_130, primals_131, primals_133, primals_139, primals_140, primals_142, primals_148, primals_149, buf4, primals_153, primals_155, primals_161, primals_162, primals_164, primals_170, primals_171, primals_173, primals_179, buf6, buf8, buf10, buf11, buf13, buf14, buf17, reinterpret_tensor(buf18, (1024, 96), (96, 1), 0), buf19, reinterpret_tensor(buf20, (1024, 384), (384, 1), 0), buf21, buf22, buf24, buf25, buf28, reinterpret_tensor(buf29, (1024, 96), (96, 1), 0), buf30, buf31, reinterpret_tensor(buf32, (1024, 384), (384, 1), 0), buf33, buf34, buf36, buf37, buf40, reinterpret_tensor(buf41, (1024, 96), (96, 1), 0), buf42, buf43, reinterpret_tensor(buf44, (1024, 384), (384, 1), 0), buf45, buf47, buf49, buf50, buf52, buf54, buf55, buf58, reinterpret_tensor(buf59, (256, 192), (192, 1), 0), buf60, buf61, reinterpret_tensor(buf62, (256, 768), (768, 1), 0), buf63, buf64, buf66, buf67, buf70, reinterpret_tensor(buf71, (256, 192), (192, 1), 0), buf72, buf73, reinterpret_tensor(buf74, (256, 768), (768, 1), 0), buf75, buf76, buf78, buf79, buf82, reinterpret_tensor(buf83, (256, 192), (192, 1), 0), buf84, buf85, reinterpret_tensor(buf86, (256, 768), (768, 1), 0), buf87, buf88, buf91, buf94, buf95, buf97, buf99, buf100, buf103, reinterpret_tensor(buf104, (64, 384), (384, 1), 0), buf105, buf106, reinterpret_tensor(buf107, (64, 1536), (1536, 1), 0), buf108, buf109, buf111, buf112, buf115, reinterpret_tensor(buf116, (64, 384), (384, 1), 0), buf117, buf118, reinterpret_tensor(buf119, (64, 1536), (1536, 1), 0), buf120, buf121, buf123, buf124, buf127, reinterpret_tensor(buf128, (64, 384), (384, 1), 0), buf129, buf130, reinterpret_tensor(buf131, (64, 1536), (1536, 1), 0), buf132, buf133, buf135, buf136, buf139, reinterpret_tensor(buf140, (64, 384), (384, 1), 0), buf141, buf142, reinterpret_tensor(buf143, (64, 1536), (1536, 1), 0), buf144, buf145, buf147, buf148, buf151, reinterpret_tensor(buf152, (64, 384), (384, 1), 0), buf153, buf154, reinterpret_tensor(buf155, (64, 1536), (1536, 1), 0), buf156, buf157, buf159, buf160, buf163, reinterpret_tensor(buf164, (64, 384), (384, 1), 0), buf165, buf166, reinterpret_tensor(buf167, (64, 1536), (1536, 1), 0), buf168, buf169, buf171, buf172, buf175, reinterpret_tensor(buf176, (64, 384), (384, 1), 0), buf177, buf178, reinterpret_tensor(buf179, (64, 1536), (1536, 1), 0), buf180, buf181, buf183, buf184, buf187, reinterpret_tensor(buf188, (64, 384), (384, 1), 0), buf189, buf190, reinterpret_tensor(buf191, (64, 1536), (1536, 1), 0), buf192, buf193, buf195, buf196, buf199, reinterpret_tensor(buf200, (64, 384), (384, 1), 0), buf201, buf202, reinterpret_tensor(buf203, (64, 1536), (1536, 1), 0), buf204, buf205, buf208, buf211, buf212, buf214, buf216, buf217, buf220, reinterpret_tensor(buf221, (16, 768), (768, 1), 0), buf222, buf223, reinterpret_tensor(buf224, (16, 3072), (3072, 1), 0), buf225, buf226, buf228, buf229, buf232, reinterpret_tensor(buf233, (16, 768), (768, 1), 0), buf234, buf235, reinterpret_tensor(buf236, (16, 3072), (3072, 1), 0), buf237, buf238, buf240, buf241, buf244, reinterpret_tensor(buf245, (16, 768), (768, 1), 0), buf246, buf247, reinterpret_tensor(buf248, (16, 3072), (3072, 1), 0), buf249, primals_177, primals_175, primals_168, primals_166, primals_159, primals_157, primals_146, primals_144, primals_137, primals_135, primals_128, primals_126, primals_119, primals_117, primals_110, primals_108, primals_101, primals_99, primals_92, primals_90, primals_83, primals_81, primals_74, primals_72, primals_61, primals_59, primals_52, primals_50, primals_43, primals_41, primals_30, primals_28, primals_21, primals_19, primals_12, primals_10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((96, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((96, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((96, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((96, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, 96, 2, 2), (384, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, 384, 2, 2), (1536, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
