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


# kernel path: inductor_cache/vt/cvtn4kewj2x2kmjquh3i4ipxazajkqrv4yukbgifkukj3qjdzzmg.py
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
    size_hints={'y': 1024, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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


# kernel path: inductor_cache/tz/ctzbtkvkyehfxzarvl2p7gdtsmsvqiguaxydkrybi3ra7uykppmp.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zp/czp54opamiaruwjt2u7onjr3li5k5adijov7yhiy4kpgd357lr6j.py
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
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/wg/cwgf3vy4cdewwngqafqs7zr74uhhw52qvqiu4ewcp3niupxjfnm5.py
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
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ae/caendbk3kulyu4fci6prmucmnpe6ryj3v2igb4yutoqtsmbrrcj4.py
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
    size_hints={'y': 4194304, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 2048*x2 + 18432*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jo/cjox7ultkslv6gd7udb27juzrlre5bplphedjbuqg5hnonid5xsj.py
# Topologically Sorted Source Nodes: [var_mean, sub, add, sqrt, w], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add => add
#   sqrt => sqrt
#   sub => sub
#   var_mean => var_mean
#   w => div
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-10), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %sqrt), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_6 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_6(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 147
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 147*x0), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 147, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 147.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-10
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + 147*x0), tmp23, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/by/cbym5rtliwag2srj2pasnszbom5kjzrxh7555itfujkmew6orlnr.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_2 => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convolution, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_7 = async_compile.triton('triton_poi_fused_constant_pad_nd_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1183744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 8704) % 34)
    x1 = ((xindex // 256) % 34)
    x3 = xindex // 295936
    x4 = (xindex % 8704)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-8448) + x4 + 8192*x2 + 262144*x3), tmp10, other=0.0)
    tl.store(out_ptr0 + (x6), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crza2lo5q7gwygss3cfhswxb43uj7mbnvmz2dz2ljc7jwhs3vxl7.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_3 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_8 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 16)
    x2 = ((xindex // 4096) % 16)
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp5 = tl.load(in_ptr0 + (8704 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp7 = tl.load(in_ptr0 + (8960 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp9 = tl.load(in_ptr0 + (9216 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp11 = tl.load(in_ptr0 + (17408 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp13 = tl.load(in_ptr0 + (17664 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
    tmp15 = tl.load(in_ptr0 + (17920 + x0 + 512*x1 + 17408*x2 + 295936*x3), None)
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
    tl.store(out_ptr0 + (x4), tmp16, None)
    tl.store(out_ptr1 + (x4), tmp41, None)
''', device_str='cuda')


# kernel path: inductor_cache/tn/ctnmsxwv4ezjvk3gwvldvxuqpj3ez6fbpjtuxzjaqc5nqvrapa3k.py
# Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm => add_1, rsqrt, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
triton_red_fused_native_group_norm_9 = async_compile.triton('triton_red_fused_native_group_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_9(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex % 8)
        r3 = rindex // 8
        tmp0 = tl.load(in_ptr0 + (r2 + 8*x0 + 256*r3 + 65536*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zk/czknfxtyqxgfvyidrdtebmmkir7pf2mveg4tlzq2bq3bnl45j3tj.py
# Topologically Sorted Source Nodes: [group_norm, out], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm => add_2, mul_1
#   out => relu
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_2,), kwargs = {})
triton_poi_fused_native_group_norm_relu_10 = async_compile.triton('triton_poi_fused_native_group_norm_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxq3v3d3dlcm53nvspycbe4wlo5anydwj5t6psmlamp4j7hdeki.py
# Topologically Sorted Source Nodes: [var_mean_1, sub_1, add_1, sqrt_1, w_1], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_1 => add_3
#   sqrt_1 => sqrt_1
#   sub_1 => sub_2
#   var_mean_1 => var_mean_2
#   w_1 => div_1
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_5, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_5, %getitem_7), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-10), kwargs = {})
#   %sqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_3,), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %sqrt_1), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_11 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_11(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-10
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + 256*x0), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/3v/c3vxdde6aqjlc77ra5pgn4zdof7rykvx545qpzqroj3icvdg3hzy.py
# Topologically Sorted Source Nodes: [var_mean_2, sub_2, add_2, sqrt_2, w_2], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_2 => add_4
#   sqrt_2 => sqrt_2
#   sub_2 => sub_3
#   var_mean_2 => var_mean_3
#   w_2 => div_2
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_6, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_6, %getitem_9), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-10), kwargs = {})
#   %sqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, %sqrt_2), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_12 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_12(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-10
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + 256*x0), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/3c/c3cc4n22ixxizywfrfy6cd2i36gsbvod2jvazj5mobjpg524blgh.py
# Topologically Sorted Source Nodes: [var_mean_3, sub_3, add_3, sqrt_3, w_3], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_3 => add_7
#   sqrt_3 => sqrt_3
#   sub_3 => sub_5
#   var_mean_3 => var_mean_5
#   w_3 => div_3
# Graph fragment:
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_9, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_9, %getitem_13), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-10), kwargs = {})
#   %sqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_7,), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, %sqrt_3), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_13 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_13(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2304*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 2304.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 2304*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 2304*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jb/cjbuaplhivytaggchzohdzs6znwkuctbb6tvgnqfyzrw7yzleti2.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_4 => add_11
# Graph fragment:
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_4, %convolution_1), kwargs = {})
triton_poi_fused_add_14 = async_compile.triton('triton_poi_fused_add_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6b/c6b77gje4r4qdcbp244tuwolxmlfabopluf7hgyigy244bv7fmk4.py
# Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_3 => add_12, rsqrt_3, var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_6, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
triton_red_fused_native_group_norm_15 = async_compile.triton('triton_red_fused_native_group_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_15(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex % 32)
        r3 = rindex // 32
        tmp0 = tl.load(in_ptr0 + (r2 + 32*x0 + 1024*r3 + 262144*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tmp5 = 8192.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ym/cymnietlwlhkyeiep2jubz22iq7lvozam74fudz2xhway3y2neu2.py
# Topologically Sorted Source Nodes: [group_norm_3, out_4], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_3 => add_13, mul_7
#   out_4 => relu_3
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %unsqueeze_23), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_20), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused_native_group_norm_relu_16 = async_compile.triton('triton_poi_fused_native_group_norm_relu_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1024)
    x2 = xindex // 262144
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/nr/cnry4oixipvmcxixut6qyye7laoklsbuanku5kuxgzqdlf7n42p6.py
# Topologically Sorted Source Nodes: [var_mean_5, sub_5, add_6, sqrt_5, w_5], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_6 => add_14
#   sqrt_5 => sqrt_5
#   sub_5 => sub_9
#   var_mean_5 => var_mean_9
#   w_5 => div_5
# Graph fragment:
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_15, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_15, %getitem_21), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-10), kwargs = {})
#   %sqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_9, %sqrt_5), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_17 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_17(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-10
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + 1024*x0), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/v7/cv7m3htg64xu7wz3za5sveisdr4gqtgexsdzxtgmektfjj6pzd4x.py
# Topologically Sorted Source Nodes: [var_mean_14, sub_14, add_18, sqrt_14, w_14], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_18 => add_44
#   sqrt_14 => sqrt_14
#   sub_14 => sub_27
#   var_mean_14 => var_mean_27
#   w_14 => div_14
# Graph fragment:
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_42, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_42, %getitem_57), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_56, 1e-10), kwargs = {})
#   %sqrt_14 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_44,), kwargs = {})
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_27, %sqrt_14), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_18 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_18(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-10
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + 1024*x0), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/p6/cp6cuilpjuqa32ngqysidobj7lnt4fg2j4ek5eaj3f3u462m6xma.py
# Topologically Sorted Source Nodes: [var_mean_15, sub_15, add_19, sqrt_15, w_15], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_19 => add_45
#   sqrt_15 => sqrt_15
#   sub_15 => sub_28
#   var_mean_15 => var_mean_28
#   w_15 => div_15
# Graph fragment:
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_43, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_43, %getitem_59), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-10), kwargs = {})
#   %sqrt_15 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_45,), kwargs = {})
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_28, %sqrt_15), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_19 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_19(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-10
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + 1024*x0), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/aa/caanpv4wj6homnvnh5dkj5z3p4bi4v5rwrggoh4uv3o7ail2thxp.py
# Topologically Sorted Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_13 => add_46, rsqrt_13, var_mean_29
# Graph fragment:
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_26, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_60, 1e-05), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_46,), kwargs = {})
triton_red_fused_native_group_norm_20 = async_compile.triton('triton_red_fused_native_group_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_20(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex % 16)
        r3 = rindex // 16
        tmp0 = tl.load(in_ptr0 + (r2 + 16*x0 + 512*r3 + 131072*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tmp5 = 4096.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dy/cdy7a7fktncdtcmmlcq6zducwdmve56nt7ue2ibpnfz7xoeysqsh.py
# Topologically Sorted Source Nodes: [group_norm_13, relu_13], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_13 => add_47, mul_27
#   relu_13 => relu_13
# Graph fragment:
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %unsqueeze_83), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_80), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
triton_poi_fused_native_group_norm_relu_21 = async_compile.triton('triton_poi_fused_native_group_norm_relu_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 131072
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4096.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxngyaljmqanp3wpok7647nbeh7xolmtagngmsoslii3tyrqhy7.py
# Topologically Sorted Source Nodes: [var_mean_16, sub_16, add_20, sqrt_16, w_16], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_20 => add_48
#   sqrt_16 => sqrt_16
#   sub_16 => sub_30
#   var_mean_16 => var_mean_30
#   w_16 => div_16
# Graph fragment:
#   %var_mean_30 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_46, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_46, %getitem_63), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_62, 1e-10), kwargs = {})
#   %sqrt_16 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_48,), kwargs = {})
#   %div_16 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_30, %sqrt_16), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_22 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_22(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4608*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 4608.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 4608*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 4608*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xn/cxnhup6i4ua7chulpxrt2bfaifzf2cjtbxss2abiimupi4bejgwq.py
# Topologically Sorted Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_14 => add_49, rsqrt_14, var_mean_31
# Graph fragment:
#   %var_mean_31 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_64, 1e-05), kwargs = {})
#   %rsqrt_14 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
triton_per_fused_native_group_norm_23 = async_compile.triton('triton_per_fused_native_group_norm_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_23(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = (rindex % 16)
    r3 = rindex // 16
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 16*x0 + 512*r3 + 32768*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x4), tmp18, None)
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4r4cmztpenubiztjjpefapeagxqq2rdjwoixvzldr4w6gjvrbu6.py
# Topologically Sorted Source Nodes: [group_norm_14, relu_14], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_14 => add_50, mul_29
#   relu_14 => relu_14
# Graph fragment:
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %unsqueeze_89), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_86), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_50,), kwargs = {})
triton_poi_fused_native_group_norm_relu_24 = async_compile.triton('triton_poi_fused_native_group_norm_relu_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/mp/cmpdw5jgejg5xnbjbdlxayozt3ahc4avtqoayj72pgdefo5djavb.py
# Topologically Sorted Source Nodes: [var_mean_17, sub_17, add_21, sqrt_17, w_17], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_21 => add_51
#   sqrt_17 => sqrt_17
#   sub_17 => sub_32
#   var_mean_17 => var_mean_32
#   w_17 => div_17
# Graph fragment:
#   %var_mean_32 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_49, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_49, %getitem_67), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_66, 1e-10), kwargs = {})
#   %sqrt_17 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_51,), kwargs = {})
#   %div_17 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_32, %sqrt_17), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_25 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_25(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-10
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + 512*x0), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/w5/cw5t2rjiswknaa6mntcq2zaocspwbzbafyirhza2rdcnxcknb2bb.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_8 => add_52
# Graph fragment:
#   %add_52 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %convolution_14), kwargs = {})
triton_poi_fused_add_26 = async_compile.triton('triton_poi_fused_add_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/bz/cbzbismq7xceuu4iduz5xv3jumhzu3ps52wfhomp4lkhbuxbois4.py
# Topologically Sorted Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_15 => add_53, rsqrt_15, var_mean_33
# Graph fragment:
#   %var_mean_33 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_68, 1e-05), kwargs = {})
#   %rsqrt_15 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_53,), kwargs = {})
triton_red_fused_native_group_norm_27 = async_compile.triton('triton_red_fused_native_group_norm_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_27(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex % 64)
        r3 = rindex // 64
        tmp0 = tl.load(in_ptr0 + (r2 + 64*x0 + 2048*r3 + 131072*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tmp5 = 4096.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ys/cyslwnq4y52f6pkhcppmindp7m45yubvu63p3ipnz66zuyslozuc.py
# Topologically Sorted Source Nodes: [group_norm_15, out_20], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_15 => add_54, mul_31
#   out_20 => relu_15
# Graph fragment:
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_31, %unsqueeze_95), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_92), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_54,), kwargs = {})
triton_poi_fused_native_group_norm_relu_28 = async_compile.triton('triton_poi_fused_native_group_norm_relu_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 2048)
    x2 = xindex // 131072
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 64)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 64)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4096.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/jc/cjctq7bdmjpemdpkrrlharkjkzwm265pp6vcb7dsem654coffwer.py
# Topologically Sorted Source Nodes: [var_mean_18, sub_18, add_23, sqrt_18, w_18], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_23 => add_55
#   sqrt_18 => sqrt_18
#   sub_18 => sub_34
#   var_mean_18 => var_mean_34
#   w_18 => div_18
# Graph fragment:
#   %var_mean_34 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_52, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_52, %getitem_71), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_70, 1e-10), kwargs = {})
#   %sqrt_18 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_55,), kwargs = {})
#   %div_18 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_34, %sqrt_18), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_29 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_29(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 2048*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m3/cm3v24p2cde6oh4x2dyqz3e6foza7esuyuqr3bcdeiwjcui3ptuj.py
# Topologically Sorted Source Nodes: [var_mean_27, sub_27, add_35, sqrt_27, w_27], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_35 => add_85
#   sqrt_27 => sqrt_27
#   sub_27 => sub_52
#   var_mean_27 => var_mean_52
#   w_27 => div_27
# Graph fragment:
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_79, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_79, %getitem_107), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_106, 1e-10), kwargs = {})
#   %sqrt_27 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_85,), kwargs = {})
#   %div_27 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_52, %sqrt_27), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_30 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_30(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 2048*x0), tmp12, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/rs/crsnpgp3d7hz3frz6pxkgdkjhincu3qifwqfko456s76l3tt7ddi.py
# Topologically Sorted Source Nodes: [var_mean_28, sub_28, add_36, sqrt_28, w_28], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_36 => add_86
#   sqrt_28 => sqrt_28
#   sub_28 => sub_53
#   var_mean_28 => var_mean_53
#   w_28 => div_28
# Graph fragment:
#   %var_mean_53 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_80, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_80, %getitem_109), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_108, 1e-10), kwargs = {})
#   %sqrt_28 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_86,), kwargs = {})
#   %div_28 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_53, %sqrt_28), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_31 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_31(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 2048*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uw/cuwpdsqrqm5ywbxhxccdgmdbvkbwpa5rgcyt4ikmuevx2pblgzsm.py
# Topologically Sorted Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_25 => add_87, rsqrt_25, var_mean_54
# Graph fragment:
#   %var_mean_54 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_110, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_87,), kwargs = {})
triton_red_fused_native_group_norm_32 = async_compile.triton('triton_red_fused_native_group_norm_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_32(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex % 32)
        r3 = rindex // 32
        tmp0 = tl.load(in_ptr0 + (r2 + 32*x0 + 1024*r3 + 65536*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gj/cgj7fk5d6tyifbqgmbgmagh47ikbbcvoy6kda4diqb2ul554zxdv.py
# Topologically Sorted Source Nodes: [group_norm_25, relu_25], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_25 => add_88, mul_51
#   relu_25 => relu_25
# Graph fragment:
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_51, %unsqueeze_155), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %unsqueeze_152), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_88,), kwargs = {})
triton_poi_fused_native_group_norm_relu_33 = async_compile.triton('triton_poi_fused_native_group_norm_relu_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1024)
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/cs/ccsaatqdgbpvgtz4ab334uztm3ksx6ixrhlesgfda4jtnuyh3l3f.py
# Topologically Sorted Source Nodes: [var_mean_29, sub_29, add_37, sqrt_29, w_29], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_37 => add_89
#   sqrt_29 => sqrt_29
#   sub_29 => sub_55
#   var_mean_29 => var_mean_55
#   w_29 => div_29
# Graph fragment:
#   %var_mean_55 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_83, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_83, %getitem_113), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_112, 1e-10), kwargs = {})
#   %sqrt_29 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_89,), kwargs = {})
#   %div_29 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_55, %sqrt_29), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_34 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_34(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 9216*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 9216.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 9216*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 9216*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zr/czr6ipzpammcf4xmhdeq5hri5hv7tvkpkxobbbmuzya6ddoacmep.py
# Topologically Sorted Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_26 => add_90, rsqrt_26, var_mean_56
# Graph fragment:
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_52, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_114, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_90,), kwargs = {})
triton_per_fused_native_group_norm_35 = async_compile.triton('triton_per_fused_native_group_norm_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_35(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = (rindex % 32)
    r3 = rindex // 32
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 32*x0 + 1024*r3 + 16384*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x4), tmp18, None)
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/vn/cvn26gyx74jilsgryg4epluwrdvdcg4p2nlbgqz22tuul6jvn2sa.py
# Topologically Sorted Source Nodes: [group_norm_26, relu_26], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_26 => add_91, mul_53
#   relu_26 => relu_26
# Graph fragment:
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_53, %unsqueeze_161), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_158), kwargs = {})
#   %relu_26 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
triton_poi_fused_native_group_norm_relu_36 = async_compile.triton('triton_poi_fused_native_group_norm_relu_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1024)
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/dw/cdwvjngmxr5lzuzkotaa5ty7il6obaco2ysqb7loftssoc2aiaza.py
# Topologically Sorted Source Nodes: [var_mean_30, sub_30, add_38, sqrt_30, w_30], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_38 => add_92
#   sqrt_30 => sqrt_30
#   sub_30 => sub_57
#   var_mean_30 => var_mean_57
#   w_30 => div_30
# Graph fragment:
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_86, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_86, %getitem_117), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_116, 1e-10), kwargs = {})
#   %sqrt_30 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_92,), kwargs = {})
#   %div_30 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_57, %sqrt_30), kwargs = {})
triton_per_fused_add_div_sqrt_sub_var_mean_37 = async_compile.triton('triton_per_fused_add_div_sqrt_sub_var_mean_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_sqrt_sub_var_mean_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_sqrt_sub_var_mean_37(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-10
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r1 + 1024*x0), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/3h/c3h5t2muljvklzipjyxe5sns5fcsbjnjkxz4jwa7qxl7kluqe7xi.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_12 => add_93
# Graph fragment:
#   %add_93 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_27), kwargs = {})
triton_poi_fused_add_38 = async_compile.triton('triton_poi_fused_add_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_38(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/b5/cb5hta72262ova4ylrcxwqqlkregk7gz7bx7jmn3dskk7ojvny6y.py
# Topologically Sorted Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_27 => add_94, rsqrt_27, var_mean_58
# Graph fragment:
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_54, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_118, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_94,), kwargs = {})
triton_red_fused_native_group_norm_39 = async_compile.triton('triton_red_fused_native_group_norm_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_39(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex % 128)
        r3 = rindex // 128
        tmp0 = tl.load(in_ptr0 + (r2 + 128*x0 + 4096*r3 + 65536*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kv/ckvc7penhgx62l4pe7s62d7lhtv6wev52xlqb6zxsyz3qs664uvb.py
# Topologically Sorted Source Nodes: [group_norm_27, out_36], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_27 => add_95, mul_55
#   out_36 => relu_27
# Graph fragment:
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_55, %unsqueeze_167), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %unsqueeze_164), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_95,), kwargs = {})
triton_poi_fused_native_group_norm_relu_40 = async_compile.triton('triton_poi_fused_native_group_norm_relu_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 4096)
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 128)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 128)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/yj/cyjce6363mm6zqiohwot2nf7rbl75sm4u4lqwdlg2l3677re4oqd.py
# Topologically Sorted Source Nodes: [var_mean_31, sub_31, add_40, sqrt_31, w_31], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_40 => add_96
#   sqrt_31 => sqrt_31
#   sub_31 => sub_59
#   var_mean_31 => var_mean_59
#   w_31 => div_31
# Graph fragment:
#   %var_mean_59 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_89, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_89, %getitem_121), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_120, 1e-10), kwargs = {})
#   %sqrt_31 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_96,), kwargs = {})
#   %div_31 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_59, %sqrt_31), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_41 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_41(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 4096.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 4096*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c5/cc5kzdi76zhgedhyjosqicuxrvgex4dmxwuo7naqkk4rigrijw75.py
# Topologically Sorted Source Nodes: [var_mean_40, sub_40, add_52, sqrt_40, w_40], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_52 => add_126
#   sqrt_40 => sqrt_40
#   sub_40 => sub_77
#   var_mean_40 => var_mean_77
#   w_40 => div_40
# Graph fragment:
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_116, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_116, %getitem_157), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_156, 1e-10), kwargs = {})
#   %sqrt_40 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_126,), kwargs = {})
#   %div_40 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_77, %sqrt_40), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_42 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_42(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 4096.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 4096*x0), tmp12, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/7h/c7h2be2qykmxenerxfsdirqt2yt33yvw2bl2mtyyamtrom4ducph.py
# Topologically Sorted Source Nodes: [var_mean_41, sub_41, add_53, sqrt_41, w_41], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_53 => add_127
#   sqrt_41 => sqrt_41
#   sub_41 => sub_78
#   var_mean_41 => var_mean_78
#   w_41 => div_41
# Graph fragment:
#   %var_mean_78 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_117, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_117, %getitem_159), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_158, 1e-10), kwargs = {})
#   %sqrt_41 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_127,), kwargs = {})
#   %div_41 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_78, %sqrt_41), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_43 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_43(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 4096.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 4096*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/od/cod2czj63lzsrdmg4bpygyfvm5j5goh2wcufugrb4nyow6jupvbo.py
# Topologically Sorted Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_37 => add_128, rsqrt_37, var_mean_79
# Graph fragment:
#   %var_mean_79 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_74, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_160, 1e-05), kwargs = {})
#   %rsqrt_37 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_128,), kwargs = {})
triton_per_fused_native_group_norm_44 = async_compile.triton('triton_per_fused_native_group_norm_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_44(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = (rindex % 64)
    r3 = rindex // 64
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 64*x0 + 2048*r3 + 32768*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x4), tmp18, None)
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/yk/cykoszkz264olkg7tje2z2g766yzskapgpzjyvjr465gymusisru.py
# Topologically Sorted Source Nodes: [group_norm_37, relu_37], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_37 => add_129, mul_75
#   relu_37 => relu_37
# Graph fragment:
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_75, %unsqueeze_227), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %unsqueeze_224), kwargs = {})
#   %relu_37 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_129,), kwargs = {})
triton_poi_fused_native_group_norm_relu_45 = async_compile.triton('triton_poi_fused_native_group_norm_relu_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 2048)
    x2 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 64)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 64)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/sg/csgpzk3t3sjgai4rmf73wjrqgzeouzwh5ivfrkri4rckgiefevzy.py
# Topologically Sorted Source Nodes: [var_mean_42, sub_42, add_54, sqrt_42, w_42], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_54 => add_130
#   sqrt_42 => sqrt_42
#   sub_42 => sub_80
#   var_mean_42 => var_mean_80
#   w_42 => div_42
# Graph fragment:
#   %var_mean_80 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_120, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_120, %getitem_163), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_162, 1e-10), kwargs = {})
#   %sqrt_42 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_130,), kwargs = {})
#   %div_42 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_80, %sqrt_42), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_46 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_46(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 18432*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 18432.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 18432*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 18432*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/so/csocoxdgokim5ht23dw376po3tintjobte6pj6x6ojzl6ejprcei.py
# Topologically Sorted Source Nodes: [group_norm_38], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_38 => add_131, rsqrt_38, var_mean_81
# Graph fragment:
#   %var_mean_81 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_164, 1e-05), kwargs = {})
#   %rsqrt_38 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_131,), kwargs = {})
triton_per_fused_native_group_norm_47 = async_compile.triton('triton_per_fused_native_group_norm_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_47(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = (rindex % 64)
    r3 = rindex // 64
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 64*x0 + 2048*r3 + 8192*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x4), tmp18, None)
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/sm/csmmi2mmqeaococj2ujvmbdchr63l3qcvz3a65iajmp7cbtvr4qx.py
# Topologically Sorted Source Nodes: [group_norm_38, relu_38], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_38 => add_132, mul_77
#   relu_38 => relu_38
# Graph fragment:
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_77, %unsqueeze_233), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_230), kwargs = {})
#   %relu_38 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_132,), kwargs = {})
triton_poi_fused_native_group_norm_relu_48 = async_compile.triton('triton_poi_fused_native_group_norm_relu_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 2048)
    x2 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 64)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 64)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/tp/ctpatadne7cyvyacabqyyheog3fu2empddsvk5wfymv2n2v6ebrj.py
# Topologically Sorted Source Nodes: [var_mean_43, sub_43, add_55, sqrt_43, w_43], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_55 => add_133
#   sqrt_43 => sqrt_43
#   sub_43 => sub_82
#   var_mean_43 => var_mean_82
#   w_43 => div_43
# Graph fragment:
#   %var_mean_82 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_123, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_123, %getitem_167), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_166, 1e-10), kwargs = {})
#   %sqrt_43 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_133,), kwargs = {})
#   %div_43 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_82, %sqrt_43), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_49 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_49(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 2048*x0), tmp12, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/tf/ctfv4ylhnfjukzwcls6ikdvdvoyg2y2w6ragcq5r5dkgj4wscone.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_16 => add_134
# Graph fragment:
#   %add_134 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_43, %convolution_40), kwargs = {})
triton_poi_fused_add_50 = async_compile.triton('triton_poi_fused_add_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_50(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/c5/cc5ddqfk6zugvxesgw7igxdihrqbma57oew3pshkixyusfgykdbo.py
# Topologically Sorted Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_39 => add_135, rsqrt_39, var_mean_83
# Graph fragment:
#   %var_mean_83 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_78, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_168, 1e-05), kwargs = {})
#   %rsqrt_39 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_135,), kwargs = {})
triton_per_fused_native_group_norm_51 = async_compile.triton('triton_per_fused_native_group_norm_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_51(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = (rindex % 256)
    r3 = rindex // 256
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 256*x0 + 8192*r3 + 32768*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x4), tmp18, None)
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/7r/c7rs2m7vdzieanqf5k6dwud6kwi5xoafoksa3aehesji5gmef34u.py
# Topologically Sorted Source Nodes: [group_norm_39, out_52], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_39 => add_136, mul_79
#   out_52 => relu_39
# Graph fragment:
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_79, %unsqueeze_239), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %unsqueeze_236), kwargs = {})
#   %relu_39 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_136,), kwargs = {})
triton_poi_fused_native_group_norm_relu_52 = async_compile.triton('triton_poi_fused_native_group_norm_relu_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 8192)
    x2 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x2 + (x0 // 256)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32*x2 + (x0 // 256)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/rj/crjpwftxw5rc4wvceipvys33r3sj6b3pfzvwgc6ku3azhkp5hzh6.py
# Topologically Sorted Source Nodes: [var_mean_44, sub_44, add_57, sqrt_44, w_44], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   add_57 => add_137
#   sqrt_44 => sqrt_44
#   sub_44 => sub_84
#   var_mean_44 => var_mean_84
#   w_44 => div_44
# Graph fragment:
#   %var_mean_84 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_126, [1, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_126, %getitem_171), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_170, 1e-10), kwargs = {})
#   %sqrt_44 : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_137,), kwargs = {})
#   %div_44 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_84, %sqrt_44), kwargs = {})
triton_red_fused_add_div_sqrt_sub_var_mean_53 = async_compile.triton('triton_red_fused_add_div_sqrt_sub_var_mean_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_sqrt_sub_var_mean_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_sqrt_sub_var_mean_53(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = 8192.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-10
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tmp11 / tmp9
        tl.store(out_ptr1 + (r1 + 8192*x0), tmp12, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/en/cenrb6ff4sldi4i5ugz2gxyyf6n74mdnptvhzitj7tp7sibyelyu.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_20 => add_165, rsqrt_48, var_mean_101
# Graph fragment:
#   %var_mean_101 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_96, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_204, 1e-05), kwargs = {})
#   %rsqrt_48 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_165,), kwargs = {})
triton_per_fused_native_group_norm_54 = async_compile.triton('triton_per_fused_native_group_norm_54', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_54(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = (rindex % 256)
    r3 = rindex // 256
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 256*x0 + 8192*r3 + 32768*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp18, None)
    tl.store(out_ptr0 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/ig/cigx3leyevmb7a2fgbhqepbbrheyggmwnlalajrftdnydlzfx5iw.py
# Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten.native_group_norm, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   input_20 => add_166, mul_97
#   input_21 => relu_48
#   input_22 => mean
# Graph fragment:
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_97, %unsqueeze_293), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %unsqueeze_290), kwargs = {})
#   %relu_48 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_166,), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_native_group_norm_relu_55 = async_compile.triton('triton_poi_fused_mean_native_group_norm_relu_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_native_group_norm_relu_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_native_group_norm_relu_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32768*x1), None)
    tmp1 = tl.load(in_ptr1 + (x2 // 256), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 // 256), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (8192 + x0 + 32768*x1), None)
    tmp18 = tl.load(in_ptr0 + (16384 + x0 + 32768*x1), None)
    tmp25 = tl.load(in_ptr0 + (24576 + x0 + 32768*x1), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = tmp11 - tmp1
    tmp13 = tmp12 * tmp3
    tmp14 = tmp13 * tmp5
    tmp15 = tmp14 + tmp7
    tmp16 = triton_helpers.maximum(tmp9, tmp15)
    tmp17 = tmp10 + tmp16
    tmp19 = tmp18 - tmp1
    tmp20 = tmp19 * tmp3
    tmp21 = tmp20 * tmp5
    tmp22 = tmp21 + tmp7
    tmp23 = triton_helpers.maximum(tmp9, tmp22)
    tmp24 = tmp17 + tmp23
    tmp26 = tmp25 - tmp1
    tmp27 = tmp26 * tmp3
    tmp28 = tmp27 * tmp5
    tmp29 = tmp28 + tmp7
    tmp30 = triton_helpers.maximum(tmp9, tmp29)
    tmp31 = tmp24 + tmp30
    tmp32 = 4.0
    tmp33 = tmp31 / tmp32
    tl.store(out_ptr0 + (x2), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/4j/c4jqgbtfxusma2q3fwc7vd3c32mxqjvqbz23shhiaqomwe7meezu.py
# Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_23 => convolution_53
# Graph fragment:
#   %convolution_53 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_153, %primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_56 = async_compile.triton('triton_poi_fused_convolution_56', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_56(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87372
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 21843)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154 = args
    args.clear()
    assert_size_stride(primals_1, (256, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_6, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_7, (256, ), (1, ))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_15, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_22, (1024, ), (1, ))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_25, (256, ), (1, ))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_31, (1024, ), (1, ))
    assert_size_stride(primals_32, (1024, ), (1, ))
    assert_size_stride(primals_33, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_40, (1024, ), (1, ))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_42, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_43, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_47, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_50, (2048, ), (1, ))
    assert_size_stride(primals_51, (2048, ), (1, ))
    assert_size_stride(primals_52, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (2048, ), (1, ))
    assert_size_stride(primals_60, (2048, ), (1, ))
    assert_size_stride(primals_61, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_68, (2048, ), (1, ))
    assert_size_stride(primals_69, (2048, ), (1, ))
    assert_size_stride(primals_70, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_77, (2048, ), (1, ))
    assert_size_stride(primals_78, (2048, ), (1, ))
    assert_size_stride(primals_79, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_80, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, ), (1, ))
    assert_size_stride(primals_83, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (1024, ), (1, ))
    assert_size_stride(primals_86, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_87, (4096, ), (1, ))
    assert_size_stride(primals_88, (4096, ), (1, ))
    assert_size_stride(primals_89, (1024, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_90, (1024, ), (1, ))
    assert_size_stride(primals_91, (1024, ), (1, ))
    assert_size_stride(primals_92, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (1024, ), (1, ))
    assert_size_stride(primals_95, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_96, (4096, ), (1, ))
    assert_size_stride(primals_97, (4096, ), (1, ))
    assert_size_stride(primals_98, (1024, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_99, (1024, ), (1, ))
    assert_size_stride(primals_100, (1024, ), (1, ))
    assert_size_stride(primals_101, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_102, (1024, ), (1, ))
    assert_size_stride(primals_103, (1024, ), (1, ))
    assert_size_stride(primals_104, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_105, (4096, ), (1, ))
    assert_size_stride(primals_106, (4096, ), (1, ))
    assert_size_stride(primals_107, (1024, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_109, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (1024, ), (1, ))
    assert_size_stride(primals_113, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_114, (4096, ), (1, ))
    assert_size_stride(primals_115, (4096, ), (1, ))
    assert_size_stride(primals_116, (8192, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_117, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_118, (2048, ), (1, ))
    assert_size_stride(primals_119, (2048, ), (1, ))
    assert_size_stride(primals_120, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_121, (2048, ), (1, ))
    assert_size_stride(primals_122, (2048, ), (1, ))
    assert_size_stride(primals_123, (8192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_124, (8192, ), (1, ))
    assert_size_stride(primals_125, (8192, ), (1, ))
    assert_size_stride(primals_126, (2048, 8192, 1, 1), (8192, 1, 1, 1))
    assert_size_stride(primals_127, (2048, ), (1, ))
    assert_size_stride(primals_128, (2048, ), (1, ))
    assert_size_stride(primals_129, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_130, (2048, ), (1, ))
    assert_size_stride(primals_131, (2048, ), (1, ))
    assert_size_stride(primals_132, (8192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_133, (8192, ), (1, ))
    assert_size_stride(primals_134, (8192, ), (1, ))
    assert_size_stride(primals_135, (2048, 8192, 1, 1), (8192, 1, 1, 1))
    assert_size_stride(primals_136, (2048, ), (1, ))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_138, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_139, (2048, ), (1, ))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (8192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_142, (8192, ), (1, ))
    assert_size_stride(primals_143, (8192, ), (1, ))
    assert_size_stride(primals_144, (2048, 8192, 1, 1), (8192, 1, 1, 1))
    assert_size_stride(primals_145, (2048, ), (1, ))
    assert_size_stride(primals_146, (2048, ), (1, ))
    assert_size_stride(primals_147, (2048, 2048, 3, 3), (18432, 9, 3, 1))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (8192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_151, (8192, ), (1, ))
    assert_size_stride(primals_152, (8192, ), (1, ))
    assert_size_stride(primals_153, (21843, 8192, 1, 1), (8192, 1, 1, 1))
    assert_size_stride(primals_154, (21843, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 768, 49, grid=grid(768, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_9, buf2, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_9
        buf3 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_18, buf3, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_18
        buf4 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_27, buf4, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_27
        buf5 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_36, buf5, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_36
        buf6 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_46, buf6, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_46
        buf7 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_55, buf7, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_55
        buf8 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_64, buf8, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_64
        buf9 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_73, buf9, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_73
        buf10 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_83, buf10, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_83
        buf11 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_92, buf11, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_92
        buf12 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_101, buf12, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_101
        buf13 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_110, buf13, 1048576, 9, grid=grid(1048576, 9), stream=stream0)
        del primals_110
        buf14 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_120, buf14, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_120
        buf15 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_129, buf15, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_129
        buf16 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_138, buf16, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_138
        buf17 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_147, buf17, 4194304, 9, grid=grid(4194304, 9), stream=stream0)
        del primals_147
        buf19 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf21 = reinterpret_tensor(buf19, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf19  # reuse
        buf22 = empty_strided_cuda((256, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean, sub, add, sqrt, w], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_6.run(buf21, buf0, buf22, 256, 147, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf1, buf22, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 256, 32, 32), (262144, 1, 8192, 256))
        buf24 = empty_strided_cuda((4, 256, 34, 34), (295936, 1, 8704, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_7.run(buf23, buf24, 1183744, grid=grid(1183744), stream=stream0)
        buf25 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf26 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.int8)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf24, buf25, buf26, 262144, grid=grid(262144), stream=stream0)
        buf27 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf28 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf30 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf25, buf27, buf28, buf30, 128, 2048, grid=grid(128), stream=stream0)
        buf31 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm, out], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf25, buf27, buf28, primals_3, primals_4, buf31, 262144, grid=grid(262144), stream=stream0)
        del primals_4
        buf33 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf35 = reinterpret_tensor(buf33, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf33  # reuse
        buf36 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_1, sub_1, add_1, sqrt_1, w_1], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_11.run(buf35, primals_5, buf36, 1024, 256, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [residual], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf31, buf36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf39 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf41 = reinterpret_tensor(buf39, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf39  # reuse
        buf42 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_2, sub_2, add_2, sqrt_2, w_2], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_12.run(buf41, primals_6, buf42, 256, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf31, buf42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf44 = buf28; del buf28  # reuse
        buf45 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf47 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf43, buf44, buf45, buf47, 128, 2048, grid=grid(128), stream=stream0)
        buf48 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_1, relu_1], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf43, buf44, buf45, primals_7, primals_8, buf48, 262144, grid=grid(262144), stream=stream0)
        del primals_8
        buf50 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf52 = reinterpret_tensor(buf50, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf50  # reuse
        buf53 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_3, sub_3, add_3, sqrt_3, w_3], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_13.run(buf52, buf2, buf53, 256, 2304, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf48, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf55 = buf45; del buf45  # reuse
        buf56 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf58 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf54, buf55, buf56, buf58, 128, 2048, grid=grid(128), stream=stream0)
        buf59 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_2, relu_2], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf54, buf55, buf56, primals_10, primals_11, buf59, 262144, grid=grid(262144), stream=stream0)
        del primals_11
        buf61 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf63 = reinterpret_tensor(buf61, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf61  # reuse
        buf64 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_4, sub_4, add_4, sqrt_4, w_4], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_11.run(buf63, primals_12, buf64, 1024, 256, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf59, buf64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_14.run(buf66, buf37, 1048576, grid=grid(1048576), stream=stream0)
        buf67 = buf56; del buf56  # reuse
        buf68 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf70 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_15.run(buf66, buf67, buf68, buf70, 128, 8192, grid=grid(128), stream=stream0)
        buf71 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [group_norm_3, out_4], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf66, buf67, buf68, primals_13, primals_14, buf71, 1048576, grid=grid(1048576), stream=stream0)
        del primals_14
        buf73 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf75 = reinterpret_tensor(buf73, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf73  # reuse
        buf76 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_5, sub_5, add_6, sqrt_5, w_5], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_17.run(buf75, primals_15, buf76, 256, 1024, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf71, buf76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf78 = buf68; del buf68  # reuse
        buf79 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf81 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf77, buf78, buf79, buf81, 128, 2048, grid=grid(128), stream=stream0)
        buf82 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_4, relu_4], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf77, buf78, buf79, primals_16, primals_17, buf82, 262144, grid=grid(262144), stream=stream0)
        del primals_17
        buf84 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf86 = reinterpret_tensor(buf84, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf84  # reuse
        buf87 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_6, sub_6, add_7, sqrt_6, w_6], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_13.run(buf86, buf3, buf87, 256, 2304, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf82, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf89 = buf79; del buf79  # reuse
        buf90 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf92 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf88, buf89, buf90, buf92, 128, 2048, grid=grid(128), stream=stream0)
        buf93 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_5, relu_5], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf88, buf89, buf90, primals_19, primals_20, buf93, 262144, grid=grid(262144), stream=stream0)
        del primals_20
        buf95 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf97 = reinterpret_tensor(buf95, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf95  # reuse
        buf98 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_7, sub_7, add_8, sqrt_7, w_7], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_11.run(buf97, primals_21, buf98, 1024, 256, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf93, buf98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_14.run(buf100, buf66, 1048576, grid=grid(1048576), stream=stream0)
        buf101 = buf90; del buf90  # reuse
        buf102 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf104 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_15.run(buf100, buf101, buf102, buf104, 128, 8192, grid=grid(128), stream=stream0)
        buf105 = reinterpret_tensor(buf23, (4, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [group_norm_6, out_8], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf100, buf101, buf102, primals_22, primals_23, buf105, 1048576, grid=grid(1048576), stream=stream0)
        del primals_23
        buf107 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf109 = reinterpret_tensor(buf107, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf107  # reuse
        buf110 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_8, sub_8, add_10, sqrt_8, w_8], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_17.run(buf109, primals_24, buf110, 256, 1024, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf105, buf110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf112 = buf102; del buf102  # reuse
        buf113 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf115 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf111, buf112, buf113, buf115, 128, 2048, grid=grid(128), stream=stream0)
        buf116 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_7, relu_7], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf111, buf112, buf113, primals_25, primals_26, buf116, 262144, grid=grid(262144), stream=stream0)
        del primals_26
        buf118 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf120 = reinterpret_tensor(buf118, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf118  # reuse
        buf121 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_9, sub_9, add_11, sqrt_9, w_9], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_13.run(buf120, buf4, buf121, 256, 2304, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf116, buf121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf123 = buf113; del buf113  # reuse
        buf124 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf126 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf122, buf123, buf124, buf126, 128, 2048, grid=grid(128), stream=stream0)
        buf127 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_8, relu_8], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf122, buf123, buf124, primals_28, primals_29, buf127, 262144, grid=grid(262144), stream=stream0)
        del primals_29
        buf129 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf131 = reinterpret_tensor(buf129, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf129  # reuse
        buf132 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_10, sub_10, add_12, sqrt_10, w_10], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_11.run(buf131, primals_30, buf132, 1024, 256, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf127, buf132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf134 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_14.run(buf134, buf100, 1048576, grid=grid(1048576), stream=stream0)
        buf135 = buf124; del buf124  # reuse
        buf136 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf138 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_15.run(buf134, buf135, buf136, buf138, 128, 8192, grid=grid(128), stream=stream0)
        buf139 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_9, out_12], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf134, buf135, buf136, primals_31, primals_32, buf139, 1048576, grid=grid(1048576), stream=stream0)
        del primals_32
        buf141 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf143 = reinterpret_tensor(buf141, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf141  # reuse
        buf144 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_11, sub_11, add_14, sqrt_11, w_11], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_17.run(buf143, primals_33, buf144, 256, 1024, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf139, buf144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf146 = buf136; del buf136  # reuse
        buf147 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf149 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf145, buf146, buf147, buf149, 128, 2048, grid=grid(128), stream=stream0)
        buf150 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_10, relu_10], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf145, buf146, buf147, primals_34, primals_35, buf150, 262144, grid=grid(262144), stream=stream0)
        del primals_35
        buf152 = empty_strided_cuda((256, 1, 1, 1), (1, 256, 256, 256), torch.float32)
        buf154 = reinterpret_tensor(buf152, (256, 1, 1, 1), (1, 1, 1, 1), 0); del buf152  # reuse
        buf155 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_12, sub_12, add_15, sqrt_12, w_12], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_13.run(buf154, buf5, buf155, 256, 2304, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf150, buf155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf157 = buf147; del buf147  # reuse
        buf158 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf160 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_9.run(buf156, buf157, buf158, buf160, 128, 2048, grid=grid(128), stream=stream0)
        buf161 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_11, relu_11], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf156, buf157, buf158, primals_37, primals_38, buf161, 262144, grid=grid(262144), stream=stream0)
        del primals_38
        buf163 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf165 = reinterpret_tensor(buf163, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf163  # reuse
        buf166 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_13, sub_13, add_16, sqrt_13, w_13], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_11.run(buf165, primals_39, buf166, 1024, 256, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_15], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf161, buf166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 1024, 16, 16), (262144, 1, 16384, 1024))
        buf168 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_14.run(buf168, buf134, 1048576, grid=grid(1048576), stream=stream0)
        buf169 = buf158; del buf158  # reuse
        buf170 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf172 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_15.run(buf168, buf169, buf170, buf172, 128, 8192, grid=grid(128), stream=stream0)
        buf173 = empty_strided_cuda((4, 1024, 16, 16), (262144, 1, 16384, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_12, out_16], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf168, buf169, buf170, primals_40, primals_41, buf173, 1048576, grid=grid(1048576), stream=stream0)
        del primals_41
        buf175 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf177 = reinterpret_tensor(buf175, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf175  # reuse
        buf178 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_14, sub_14, add_18, sqrt_14, w_14], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_18.run(buf177, primals_42, buf178, 2048, 1024, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [residual_1], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf173, buf178, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 2048, 8, 8), (131072, 1, 16384, 2048))
        buf181 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf183 = reinterpret_tensor(buf181, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf181  # reuse
        buf184 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_15, sub_15, add_19, sqrt_15, w_15], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_19.run(buf183, primals_43, buf184, 512, 1024, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf173, buf184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 512, 16, 16), (131072, 1, 8192, 512))
        buf186 = buf170; del buf170  # reuse
        buf187 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf189 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_20.run(buf185, buf186, buf187, buf189, 128, 4096, grid=grid(128), stream=stream0)
        buf190 = empty_strided_cuda((4, 512, 16, 16), (131072, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_13, relu_13], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_21.run(buf185, buf186, buf187, primals_44, primals_45, buf190, 524288, grid=grid(524288), stream=stream0)
        del primals_45
        buf192 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf194 = reinterpret_tensor(buf192, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf192  # reuse
        buf195 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_16, sub_16, add_20, sqrt_16, w_16], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_22.run(buf194, buf6, buf195, 512, 4608, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf190, buf195, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf197 = buf187; del buf187  # reuse
        buf198 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf200 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_23.run(buf196, buf197, buf198, buf200, 128, 1024, grid=grid(128), stream=stream0)
        buf201 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_14, relu_14], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf196, buf197, buf198, primals_47, primals_48, buf201, 131072, grid=grid(131072), stream=stream0)
        del primals_48
        buf203 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf205 = reinterpret_tensor(buf203, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf203  # reuse
        buf206 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_17, sub_17, add_21, sqrt_17, w_17], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_25.run(buf205, primals_49, buf206, 2048, 512, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf201, buf206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 2048, 8, 8), (131072, 1, 16384, 2048))
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_26.run(buf208, buf179, 524288, grid=grid(524288), stream=stream0)
        buf209 = buf198; del buf198  # reuse
        buf210 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf212 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_27.run(buf208, buf209, buf210, buf212, 128, 4096, grid=grid(128), stream=stream0)
        buf213 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [group_norm_15, out_20], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_28.run(buf208, buf209, buf210, primals_50, primals_51, buf213, 524288, grid=grid(524288), stream=stream0)
        del primals_51
        buf215 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf217 = reinterpret_tensor(buf215, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf215  # reuse
        buf218 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_18, sub_18, add_23, sqrt_18, w_18], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_29.run(buf217, primals_52, buf218, 512, 2048, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf213, buf218, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf220 = buf210; del buf210  # reuse
        buf221 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf223 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_16], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_23.run(buf219, buf220, buf221, buf223, 128, 1024, grid=grid(128), stream=stream0)
        buf224 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_16, relu_16], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf219, buf220, buf221, primals_53, primals_54, buf224, 131072, grid=grid(131072), stream=stream0)
        del primals_54
        buf226 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf228 = reinterpret_tensor(buf226, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf226  # reuse
        buf229 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_19, sub_19, add_24, sqrt_19, w_19], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_22.run(buf228, buf7, buf229, 512, 4608, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf224, buf229, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf231 = buf221; del buf221  # reuse
        buf232 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf234 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_17], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_23.run(buf230, buf231, buf232, buf234, 128, 1024, grid=grid(128), stream=stream0)
        buf235 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_17, relu_17], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf230, buf231, buf232, primals_56, primals_57, buf235, 131072, grid=grid(131072), stream=stream0)
        del primals_57
        buf237 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf239 = reinterpret_tensor(buf237, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf237  # reuse
        buf240 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_20, sub_20, add_25, sqrt_20, w_20], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_25.run(buf239, primals_58, buf240, 2048, 512, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf235, buf240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 2048, 8, 8), (131072, 1, 16384, 2048))
        buf242 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_26.run(buf242, buf208, 524288, grid=grid(524288), stream=stream0)
        buf243 = buf232; del buf232  # reuse
        buf244 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf246 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_27.run(buf242, buf243, buf244, buf246, 128, 4096, grid=grid(128), stream=stream0)
        buf247 = empty_strided_cuda((4, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_18, out_24], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_28.run(buf242, buf243, buf244, primals_59, primals_60, buf247, 524288, grid=grid(524288), stream=stream0)
        del primals_60
        buf249 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf251 = reinterpret_tensor(buf249, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf249  # reuse
        buf252 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_21, sub_21, add_27, sqrt_21, w_21], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_29.run(buf251, primals_61, buf252, 512, 2048, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_25], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf247, buf252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf254 = buf244; del buf244  # reuse
        buf255 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf257 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_23.run(buf253, buf254, buf255, buf257, 128, 1024, grid=grid(128), stream=stream0)
        buf258 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_19, relu_19], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf253, buf254, buf255, primals_62, primals_63, buf258, 131072, grid=grid(131072), stream=stream0)
        del primals_63
        buf260 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf262 = reinterpret_tensor(buf260, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf260  # reuse
        buf263 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_22, sub_22, add_28, sqrt_22, w_22], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_22.run(buf262, buf8, buf263, 512, 4608, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf258, buf263, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf265 = buf255; del buf255  # reuse
        buf266 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf268 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_20], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_23.run(buf264, buf265, buf266, buf268, 128, 1024, grid=grid(128), stream=stream0)
        buf269 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_20, relu_20], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf264, buf265, buf266, primals_65, primals_66, buf269, 131072, grid=grid(131072), stream=stream0)
        del primals_66
        buf271 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf273 = reinterpret_tensor(buf271, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf271  # reuse
        buf274 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_23, sub_23, add_29, sqrt_23, w_23], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_25.run(buf273, primals_67, buf274, 2048, 512, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf269, buf274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 2048, 8, 8), (131072, 1, 16384, 2048))
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_26.run(buf276, buf242, 524288, grid=grid(524288), stream=stream0)
        buf277 = buf266; del buf266  # reuse
        buf278 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf280 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_21], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_27.run(buf276, buf277, buf278, buf280, 128, 4096, grid=grid(128), stream=stream0)
        buf281 = empty_strided_cuda((4, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_21, out_28], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_28.run(buf276, buf277, buf278, primals_68, primals_69, buf281, 524288, grid=grid(524288), stream=stream0)
        del primals_69
        buf283 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf285 = reinterpret_tensor(buf283, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf283  # reuse
        buf286 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_24, sub_24, add_31, sqrt_24, w_24], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_29.run(buf285, primals_70, buf286, 512, 2048, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_29], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf281, buf286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf288 = buf278; del buf278  # reuse
        buf289 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf291 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_22], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_23.run(buf287, buf288, buf289, buf291, 128, 1024, grid=grid(128), stream=stream0)
        buf292 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_22, relu_22], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf287, buf288, buf289, primals_71, primals_72, buf292, 131072, grid=grid(131072), stream=stream0)
        del primals_72
        buf294 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf296 = reinterpret_tensor(buf294, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf294  # reuse
        buf297 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_25, sub_25, add_32, sqrt_25, w_25], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_22.run(buf296, buf9, buf297, 512, 4608, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf292, buf297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf299 = buf289; del buf289  # reuse
        buf300 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf302 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_23], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_23.run(buf298, buf299, buf300, buf302, 128, 1024, grid=grid(128), stream=stream0)
        buf303 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_23, relu_23], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf298, buf299, buf300, primals_74, primals_75, buf303, 131072, grid=grid(131072), stream=stream0)
        del primals_75
        buf305 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf307 = reinterpret_tensor(buf305, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf305  # reuse
        buf308 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_26, sub_26, add_33, sqrt_26, w_26], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_25.run(buf307, primals_76, buf308, 2048, 512, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf303, buf308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (4, 2048, 8, 8), (131072, 1, 16384, 2048))
        buf310 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_26.run(buf310, buf276, 524288, grid=grid(524288), stream=stream0)
        buf311 = buf300; del buf300  # reuse
        buf312 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf314 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_27.run(buf310, buf311, buf312, buf314, 128, 4096, grid=grid(128), stream=stream0)
        buf315 = empty_strided_cuda((4, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_24, out_32], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_28.run(buf310, buf311, buf312, primals_77, primals_78, buf315, 524288, grid=grid(524288), stream=stream0)
        del primals_78
        buf317 = empty_strided_cuda((4096, 1, 1, 1), (1, 4096, 4096, 4096), torch.float32)
        buf319 = reinterpret_tensor(buf317, (4096, 1, 1, 1), (1, 1, 1, 1), 0); del buf317  # reuse
        buf320 = empty_strided_cuda((4096, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_27, sub_27, add_35, sqrt_27, w_27], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_30.run(buf319, primals_79, buf320, 4096, 2048, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [residual_2], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf315, buf320, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (4, 4096, 4, 4), (65536, 1, 16384, 4096))
        buf323 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf325 = reinterpret_tensor(buf323, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf323  # reuse
        buf326 = empty_strided_cuda((1024, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_28, sub_28, add_36, sqrt_28, w_28], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_31.run(buf325, primals_80, buf326, 1024, 2048, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf315, buf326, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 1024, 8, 8), (65536, 1, 8192, 1024))
        buf328 = buf312; del buf312  # reuse
        buf329 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf331 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_32.run(buf327, buf328, buf329, buf331, 128, 2048, grid=grid(128), stream=stream0)
        buf332 = empty_strided_cuda((4, 1024, 8, 8), (65536, 1, 8192, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_25, relu_25], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_33.run(buf327, buf328, buf329, primals_81, primals_82, buf332, 262144, grid=grid(262144), stream=stream0)
        del primals_82
        buf334 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf336 = reinterpret_tensor(buf334, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf334  # reuse
        buf337 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_29, sub_29, add_37, sqrt_29, w_29], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_34.run(buf336, buf10, buf337, 1024, 9216, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf332, buf337, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf339 = buf329; del buf329  # reuse
        buf340 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf342 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf338, buf339, buf340, buf342, 128, 512, grid=grid(128), stream=stream0)
        buf343 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_26, relu_26], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_36.run(buf338, buf339, buf340, primals_84, primals_85, buf343, 65536, grid=grid(65536), stream=stream0)
        del primals_85
        buf345 = empty_strided_cuda((4096, 1, 1, 1), (1, 4096, 4096, 4096), torch.float32)
        buf347 = reinterpret_tensor(buf345, (4096, 1, 1, 1), (1, 1, 1, 1), 0); del buf345  # reuse
        buf348 = empty_strided_cuda((4096, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_30, sub_30, add_38, sqrt_30, w_30], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_37.run(buf347, primals_86, buf348, 4096, 1024, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf343, buf348, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 4096, 4, 4), (65536, 1, 16384, 4096))
        buf350 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_38.run(buf350, buf321, 262144, grid=grid(262144), stream=stream0)
        buf351 = buf340; del buf340  # reuse
        buf352 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf354 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_39.run(buf350, buf351, buf352, buf354, 128, 2048, grid=grid(128), stream=stream0)
        buf355 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [group_norm_27, out_36], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_40.run(buf350, buf351, buf352, primals_87, primals_88, buf355, 262144, grid=grid(262144), stream=stream0)
        del primals_88
        buf357 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf359 = reinterpret_tensor(buf357, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf357  # reuse
        buf360 = empty_strided_cuda((1024, 4096, 1, 1), (4096, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_31, sub_31, add_40, sqrt_31, w_31], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_41.run(buf359, primals_89, buf360, 1024, 4096, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf355, buf360, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf362 = buf352; del buf352  # reuse
        buf363 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf365 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_28], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf361, buf362, buf363, buf365, 128, 512, grid=grid(128), stream=stream0)
        buf366 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_28, relu_28], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_36.run(buf361, buf362, buf363, primals_90, primals_91, buf366, 65536, grid=grid(65536), stream=stream0)
        del primals_91
        buf368 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf370 = reinterpret_tensor(buf368, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf368  # reuse
        buf371 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_32, sub_32, add_41, sqrt_32, w_32], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_34.run(buf370, buf11, buf371, 1024, 9216, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf366, buf371, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf373 = buf363; del buf363  # reuse
        buf374 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf376 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_29], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf372, buf373, buf374, buf376, 128, 512, grid=grid(128), stream=stream0)
        buf377 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_29, relu_29], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_36.run(buf372, buf373, buf374, primals_93, primals_94, buf377, 65536, grid=grid(65536), stream=stream0)
        del primals_94
        buf379 = empty_strided_cuda((4096, 1, 1, 1), (1, 4096, 4096, 4096), torch.float32)
        buf381 = reinterpret_tensor(buf379, (4096, 1, 1, 1), (1, 1, 1, 1), 0); del buf379  # reuse
        buf382 = empty_strided_cuda((4096, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_33, sub_33, add_42, sqrt_33, w_33], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_37.run(buf381, primals_95, buf382, 4096, 1024, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [out_39], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf377, buf382, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 4096, 4, 4), (65536, 1, 16384, 4096))
        buf384 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_38.run(buf384, buf350, 262144, grid=grid(262144), stream=stream0)
        buf385 = buf374; del buf374  # reuse
        buf386 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf388 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_30], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_39.run(buf384, buf385, buf386, buf388, 128, 2048, grid=grid(128), stream=stream0)
        buf389 = empty_strided_cuda((4, 4096, 4, 4), (65536, 1, 16384, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_30, out_40], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_40.run(buf384, buf385, buf386, primals_96, primals_97, buf389, 262144, grid=grid(262144), stream=stream0)
        del primals_97
        buf391 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf393 = reinterpret_tensor(buf391, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf391  # reuse
        buf394 = empty_strided_cuda((1024, 4096, 1, 1), (4096, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_34, sub_34, add_44, sqrt_34, w_34], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_41.run(buf393, primals_98, buf394, 1024, 4096, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf389, buf394, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf396 = buf386; del buf386  # reuse
        buf397 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf399 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_31], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf395, buf396, buf397, buf399, 128, 512, grid=grid(128), stream=stream0)
        buf400 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_31, relu_31], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_36.run(buf395, buf396, buf397, primals_99, primals_100, buf400, 65536, grid=grid(65536), stream=stream0)
        del primals_100
        buf402 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf404 = reinterpret_tensor(buf402, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf402  # reuse
        buf405 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_35, sub_35, add_45, sqrt_35, w_35], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_34.run(buf404, buf12, buf405, 1024, 9216, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf400, buf405, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf407 = buf397; del buf397  # reuse
        buf408 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf410 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_32], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf406, buf407, buf408, buf410, 128, 512, grid=grid(128), stream=stream0)
        buf411 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_32, relu_32], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_36.run(buf406, buf407, buf408, primals_102, primals_103, buf411, 65536, grid=grid(65536), stream=stream0)
        del primals_103
        buf413 = empty_strided_cuda((4096, 1, 1, 1), (1, 4096, 4096, 4096), torch.float32)
        buf415 = reinterpret_tensor(buf413, (4096, 1, 1, 1), (1, 1, 1, 1), 0); del buf413  # reuse
        buf416 = empty_strided_cuda((4096, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_36, sub_36, add_46, sqrt_36, w_36], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_37.run(buf415, primals_104, buf416, 4096, 1024, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf411, buf416, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (4, 4096, 4, 4), (65536, 1, 16384, 4096))
        buf418 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_38.run(buf418, buf384, 262144, grid=grid(262144), stream=stream0)
        buf419 = buf408; del buf408  # reuse
        buf420 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf422 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_33], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_39.run(buf418, buf419, buf420, buf422, 128, 2048, grid=grid(128), stream=stream0)
        buf423 = empty_strided_cuda((4, 4096, 4, 4), (65536, 1, 16384, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_33, out_44], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_40.run(buf418, buf419, buf420, primals_105, primals_106, buf423, 262144, grid=grid(262144), stream=stream0)
        del primals_106
        buf425 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf427 = reinterpret_tensor(buf425, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf425  # reuse
        buf428 = empty_strided_cuda((1024, 4096, 1, 1), (4096, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_37, sub_37, add_48, sqrt_37, w_37], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_41.run(buf427, primals_107, buf428, 1024, 4096, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf423, buf428, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf430 = buf420; del buf420  # reuse
        buf431 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf433 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_34], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf429, buf430, buf431, buf433, 128, 512, grid=grid(128), stream=stream0)
        buf434 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_34, relu_34], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_36.run(buf429, buf430, buf431, primals_108, primals_109, buf434, 65536, grid=grid(65536), stream=stream0)
        del primals_109
        buf436 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf438 = reinterpret_tensor(buf436, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf436  # reuse
        buf439 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_38, sub_38, add_49, sqrt_38, w_38], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_34.run(buf438, buf13, buf439, 1024, 9216, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf434, buf439, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf441 = buf431; del buf431  # reuse
        buf442 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf444 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_35], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_35.run(buf440, buf441, buf442, buf444, 128, 512, grid=grid(128), stream=stream0)
        buf445 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_35, relu_35], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_36.run(buf440, buf441, buf442, primals_111, primals_112, buf445, 65536, grid=grid(65536), stream=stream0)
        del primals_112
        buf447 = empty_strided_cuda((4096, 1, 1, 1), (1, 4096, 4096, 4096), torch.float32)
        buf449 = reinterpret_tensor(buf447, (4096, 1, 1, 1), (1, 1, 1, 1), 0); del buf447  # reuse
        buf450 = empty_strided_cuda((4096, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_39, sub_39, add_50, sqrt_39, w_39], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_sqrt_sub_var_mean_37.run(buf449, primals_113, buf450, 4096, 1024, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf445, buf450, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (4, 4096, 4, 4), (65536, 1, 16384, 4096))
        buf452 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_38.run(buf452, buf418, 262144, grid=grid(262144), stream=stream0)
        buf453 = buf442; del buf442  # reuse
        buf454 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf456 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_36], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_39.run(buf452, buf453, buf454, buf456, 128, 2048, grid=grid(128), stream=stream0)
        buf457 = empty_strided_cuda((4, 4096, 4, 4), (65536, 1, 16384, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_36, out_48], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_40.run(buf452, buf453, buf454, primals_114, primals_115, buf457, 262144, grid=grid(262144), stream=stream0)
        del primals_115
        buf459 = empty_strided_cuda((8192, 1, 1, 1), (1, 8192, 8192, 8192), torch.float32)
        buf461 = reinterpret_tensor(buf459, (8192, 1, 1, 1), (1, 1, 1, 1), 0); del buf459  # reuse
        buf462 = empty_strided_cuda((8192, 4096, 1, 1), (4096, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_40, sub_40, add_52, sqrt_40, w_40], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_42.run(buf461, primals_116, buf462, 8192, 4096, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [residual_3], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf457, buf462, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 8192, 2, 2), (32768, 1, 16384, 8192))
        buf465 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf467 = reinterpret_tensor(buf465, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf465  # reuse
        buf468 = empty_strided_cuda((2048, 4096, 1, 1), (4096, 1, 4096, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_41, sub_41, add_53, sqrt_41, w_41], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_43.run(buf467, primals_117, buf468, 2048, 4096, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf457, buf468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (4, 2048, 4, 4), (32768, 1, 8192, 2048))
        buf470 = buf454; del buf454  # reuse
        buf471 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf473 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_44.run(buf469, buf470, buf471, buf473, 128, 1024, grid=grid(128), stream=stream0)
        buf474 = empty_strided_cuda((4, 2048, 4, 4), (32768, 1, 8192, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_37, relu_37], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_45.run(buf469, buf470, buf471, primals_118, primals_119, buf474, 131072, grid=grid(131072), stream=stream0)
        del primals_119
        buf476 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf478 = reinterpret_tensor(buf476, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf476  # reuse
        buf479 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_42, sub_42, add_54, sqrt_42, w_42], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_46.run(buf478, buf14, buf479, 2048, 18432, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf474, buf479, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf481 = buf471; del buf471  # reuse
        buf482 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf484 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_38], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf480, buf481, buf482, buf484, 128, 256, grid=grid(128), stream=stream0)
        buf485 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_38, relu_38], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_48.run(buf480, buf481, buf482, primals_121, primals_122, buf485, 32768, grid=grid(32768), stream=stream0)
        del primals_122
        buf487 = empty_strided_cuda((8192, 1, 1, 1), (1, 8192, 8192, 8192), torch.float32)
        buf489 = reinterpret_tensor(buf487, (8192, 1, 1, 1), (1, 1, 1, 1), 0); del buf487  # reuse
        buf490 = empty_strided_cuda((8192, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_43, sub_43, add_55, sqrt_43, w_43], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_49.run(buf489, primals_123, buf490, 8192, 2048, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [out_51], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf485, buf490, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (4, 8192, 2, 2), (32768, 1, 16384, 8192))
        buf492 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_50.run(buf492, buf463, 131072, grid=grid(131072), stream=stream0)
        buf493 = buf482; del buf482  # reuse
        buf494 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf496 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_51.run(buf492, buf493, buf494, buf496, 128, 1024, grid=grid(128), stream=stream0)
        buf497 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [group_norm_39, out_52], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_52.run(buf492, buf493, buf494, primals_124, primals_125, buf497, 131072, grid=grid(131072), stream=stream0)
        del primals_125
        buf499 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf501 = reinterpret_tensor(buf499, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf499  # reuse
        buf502 = empty_strided_cuda((2048, 8192, 1, 1), (8192, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_44, sub_44, add_57, sqrt_44, w_44], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_53.run(buf501, primals_126, buf502, 2048, 8192, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf497, buf502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf504 = buf494; del buf494  # reuse
        buf505 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf507 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_40], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf503, buf504, buf505, buf507, 128, 256, grid=grid(128), stream=stream0)
        buf508 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_40, relu_40], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_48.run(buf503, buf504, buf505, primals_127, primals_128, buf508, 32768, grid=grid(32768), stream=stream0)
        del primals_128
        buf510 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf512 = reinterpret_tensor(buf510, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf510  # reuse
        buf513 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_45, sub_45, add_58, sqrt_45, w_45], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_46.run(buf512, buf15, buf513, 2048, 18432, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf508, buf513, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf515 = buf505; del buf505  # reuse
        buf516 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf518 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_41], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf514, buf515, buf516, buf518, 128, 256, grid=grid(128), stream=stream0)
        buf519 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_41, relu_41], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_48.run(buf514, buf515, buf516, primals_130, primals_131, buf519, 32768, grid=grid(32768), stream=stream0)
        del primals_131
        buf521 = empty_strided_cuda((8192, 1, 1, 1), (1, 8192, 8192, 8192), torch.float32)
        buf523 = reinterpret_tensor(buf521, (8192, 1, 1, 1), (1, 1, 1, 1), 0); del buf521  # reuse
        buf524 = empty_strided_cuda((8192, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_46, sub_46, add_59, sqrt_46, w_46], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_49.run(buf523, primals_132, buf524, 8192, 2048, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [out_55], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf519, buf524, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (4, 8192, 2, 2), (32768, 1, 16384, 8192))
        buf526 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_50.run(buf526, buf492, 131072, grid=grid(131072), stream=stream0)
        buf527 = buf516; del buf516  # reuse
        buf528 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf530 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_42], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_51.run(buf526, buf527, buf528, buf530, 128, 1024, grid=grid(128), stream=stream0)
        buf531 = empty_strided_cuda((4, 8192, 2, 2), (32768, 1, 16384, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_42, out_56], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_52.run(buf526, buf527, buf528, primals_133, primals_134, buf531, 131072, grid=grid(131072), stream=stream0)
        del primals_134
        buf533 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf535 = reinterpret_tensor(buf533, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf533  # reuse
        buf536 = empty_strided_cuda((2048, 8192, 1, 1), (8192, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_47, sub_47, add_61, sqrt_47, w_47], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_53.run(buf535, primals_135, buf536, 2048, 8192, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf531, buf536, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf537, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf538 = buf528; del buf528  # reuse
        buf539 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf541 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_43], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf537, buf538, buf539, buf541, 128, 256, grid=grid(128), stream=stream0)
        buf542 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_43, relu_43], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_48.run(buf537, buf538, buf539, primals_136, primals_137, buf542, 32768, grid=grid(32768), stream=stream0)
        del primals_137
        buf544 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf546 = reinterpret_tensor(buf544, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf544  # reuse
        buf547 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_48, sub_48, add_62, sqrt_48, w_48], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_46.run(buf546, buf16, buf547, 2048, 18432, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_58], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf542, buf547, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf549 = buf539; del buf539  # reuse
        buf550 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf552 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_44], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf548, buf549, buf550, buf552, 128, 256, grid=grid(128), stream=stream0)
        buf553 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_44, relu_44], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_48.run(buf548, buf549, buf550, primals_139, primals_140, buf553, 32768, grid=grid(32768), stream=stream0)
        del primals_140
        buf555 = empty_strided_cuda((8192, 1, 1, 1), (1, 8192, 8192, 8192), torch.float32)
        buf557 = reinterpret_tensor(buf555, (8192, 1, 1, 1), (1, 1, 1, 1), 0); del buf555  # reuse
        buf558 = empty_strided_cuda((8192, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_49, sub_49, add_63, sqrt_49, w_49], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_49.run(buf557, primals_141, buf558, 8192, 2048, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [out_59], Original ATen: [aten.convolution]
        buf559 = extern_kernels.convolution(buf553, buf558, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf559, (4, 8192, 2, 2), (32768, 1, 16384, 8192))
        buf560 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_50.run(buf560, buf526, 131072, grid=grid(131072), stream=stream0)
        buf561 = buf550; del buf550  # reuse
        buf562 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf564 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_45], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_51.run(buf560, buf561, buf562, buf564, 128, 1024, grid=grid(128), stream=stream0)
        buf565 = empty_strided_cuda((4, 8192, 2, 2), (32768, 1, 16384, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_45, out_60], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_52.run(buf560, buf561, buf562, primals_142, primals_143, buf565, 131072, grid=grid(131072), stream=stream0)
        del primals_143
        buf567 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf569 = reinterpret_tensor(buf567, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf567  # reuse
        buf570 = empty_strided_cuda((2048, 8192, 1, 1), (8192, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_50, sub_50, add_65, sqrt_50, w_50], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_53.run(buf569, primals_144, buf570, 2048, 8192, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_61], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf565, buf570, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf572 = buf562; del buf562  # reuse
        buf573 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf575 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_46], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf571, buf572, buf573, buf575, 128, 256, grid=grid(128), stream=stream0)
        buf576 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_46, relu_46], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_48.run(buf571, buf572, buf573, primals_145, primals_146, buf576, 32768, grid=grid(32768), stream=stream0)
        del primals_146
        buf578 = empty_strided_cuda((2048, 1, 1, 1), (1, 2048, 2048, 2048), torch.float32)
        buf580 = reinterpret_tensor(buf578, (2048, 1, 1, 1), (1, 1, 1, 1), 0); del buf578  # reuse
        buf581 = empty_strided_cuda((2048, 2048, 3, 3), (18432, 1, 6144, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_51, sub_51, add_66, sqrt_51, w_51], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_46.run(buf580, buf17, buf581, 2048, 18432, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [out_62], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf576, buf581, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf583 = buf573; del buf573  # reuse
        buf584 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf586 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_47], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_47.run(buf582, buf583, buf584, buf586, 128, 256, grid=grid(128), stream=stream0)
        buf587 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_47, relu_47], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_48.run(buf582, buf583, buf584, primals_148, primals_149, buf587, 32768, grid=grid(32768), stream=stream0)
        del primals_149
        buf589 = empty_strided_cuda((8192, 1, 1, 1), (1, 8192, 8192, 8192), torch.float32)
        buf591 = reinterpret_tensor(buf589, (8192, 1, 1, 1), (1, 1, 1, 1), 0); del buf589  # reuse
        buf592 = empty_strided_cuda((8192, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean_52, sub_52, add_67, sqrt_52, w_52], Original ATen: [aten.var_mean, aten.sub, aten.add, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_div_sqrt_sub_var_mean_49.run(buf591, primals_150, buf592, 8192, 2048, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf587, buf592, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (4, 8192, 2, 2), (32768, 1, 16384, 8192))
        buf594 = buf593; del buf593  # reuse
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_50.run(buf594, buf560, 131072, grid=grid(131072), stream=stream0)
        buf595 = reinterpret_tensor(buf584, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf584  # reuse
        buf596 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf598 = reinterpret_tensor(buf596, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_54.run(buf598, buf594, buf595, 128, 1024, grid=grid(128), stream=stream0)
        buf599 = empty_strided_cuda((4, 8192, 1, 1), (8192, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten.native_group_norm, aten.relu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_native_group_norm_relu_55.run(buf594, buf595, buf598, primals_151, primals_152, buf599, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (4, 21843, 1, 1), (21843, 1, 21843, 21843))
        buf601 = reinterpret_tensor(buf600, (4, 21843, 1, 1), (21843, 1, 87372, 87372), 0); del buf600  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_56.run(buf601, primals_154, 87372, grid=grid(87372), stream=stream0)
        del primals_154
    return (reinterpret_tensor(buf601, (4, 21843), (21843, 1), 0), buf0, buf1, primals_3, primals_5, primals_6, primals_7, buf2, primals_10, primals_12, primals_13, primals_15, primals_16, buf3, primals_19, primals_21, primals_22, primals_24, primals_25, buf4, primals_28, primals_30, primals_31, primals_33, primals_34, buf5, primals_37, primals_39, primals_40, primals_42, primals_43, primals_44, buf6, primals_47, primals_49, primals_50, primals_52, primals_53, buf7, primals_56, primals_58, primals_59, primals_61, primals_62, buf8, primals_65, primals_67, primals_68, primals_70, primals_71, buf9, primals_74, primals_76, primals_77, primals_79, primals_80, primals_81, buf10, primals_84, primals_86, primals_87, primals_89, primals_90, buf11, primals_93, primals_95, primals_96, primals_98, primals_99, buf12, primals_102, primals_104, primals_105, primals_107, primals_108, buf13, primals_111, primals_113, primals_114, primals_116, primals_117, primals_118, buf14, primals_121, primals_123, primals_124, primals_126, primals_127, buf15, primals_130, primals_132, primals_133, primals_135, primals_136, buf16, primals_139, primals_141, primals_142, primals_144, primals_145, buf17, primals_148, primals_150, primals_151, primals_152, primals_153, buf21, buf22, buf24, buf25, buf26, reinterpret_tensor(buf27, (4, 32), (32, 1), 0), reinterpret_tensor(buf30, (4, 32), (32, 1), 0), buf31, buf35, buf36, buf41, buf42, buf43, reinterpret_tensor(buf44, (4, 32), (32, 1), 0), reinterpret_tensor(buf47, (4, 32), (32, 1), 0), buf48, buf52, buf53, buf54, reinterpret_tensor(buf55, (4, 32), (32, 1), 0), reinterpret_tensor(buf58, (4, 32), (32, 1), 0), buf59, buf63, buf64, buf66, reinterpret_tensor(buf67, (4, 32), (32, 1), 0), reinterpret_tensor(buf70, (4, 32), (32, 1), 0), buf71, buf75, buf76, buf77, reinterpret_tensor(buf78, (4, 32), (32, 1), 0), reinterpret_tensor(buf81, (4, 32), (32, 1), 0), buf82, buf86, buf87, buf88, reinterpret_tensor(buf89, (4, 32), (32, 1), 0), reinterpret_tensor(buf92, (4, 32), (32, 1), 0), buf93, buf97, buf98, buf100, reinterpret_tensor(buf101, (4, 32), (32, 1), 0), reinterpret_tensor(buf104, (4, 32), (32, 1), 0), buf105, buf109, buf110, buf111, reinterpret_tensor(buf112, (4, 32), (32, 1), 0), reinterpret_tensor(buf115, (4, 32), (32, 1), 0), buf116, buf120, buf121, buf122, reinterpret_tensor(buf123, (4, 32), (32, 1), 0), reinterpret_tensor(buf126, (4, 32), (32, 1), 0), buf127, buf131, buf132, buf134, reinterpret_tensor(buf135, (4, 32), (32, 1), 0), reinterpret_tensor(buf138, (4, 32), (32, 1), 0), buf139, buf143, buf144, buf145, reinterpret_tensor(buf146, (4, 32), (32, 1), 0), reinterpret_tensor(buf149, (4, 32), (32, 1), 0), buf150, buf154, buf155, buf156, reinterpret_tensor(buf157, (4, 32), (32, 1), 0), reinterpret_tensor(buf160, (4, 32), (32, 1), 0), buf161, buf165, buf166, buf168, reinterpret_tensor(buf169, (4, 32), (32, 1), 0), reinterpret_tensor(buf172, (4, 32), (32, 1), 0), buf173, buf177, buf178, buf183, buf184, buf185, reinterpret_tensor(buf186, (4, 32), (32, 1), 0), reinterpret_tensor(buf189, (4, 32), (32, 1), 0), buf190, buf194, buf195, buf196, reinterpret_tensor(buf197, (4, 32), (32, 1), 0), reinterpret_tensor(buf200, (4, 32), (32, 1), 0), buf201, buf205, buf206, buf208, reinterpret_tensor(buf209, (4, 32), (32, 1), 0), reinterpret_tensor(buf212, (4, 32), (32, 1), 0), buf213, buf217, buf218, buf219, reinterpret_tensor(buf220, (4, 32), (32, 1), 0), reinterpret_tensor(buf223, (4, 32), (32, 1), 0), buf224, buf228, buf229, buf230, reinterpret_tensor(buf231, (4, 32), (32, 1), 0), reinterpret_tensor(buf234, (4, 32), (32, 1), 0), buf235, buf239, buf240, buf242, reinterpret_tensor(buf243, (4, 32), (32, 1), 0), reinterpret_tensor(buf246, (4, 32), (32, 1), 0), buf247, buf251, buf252, buf253, reinterpret_tensor(buf254, (4, 32), (32, 1), 0), reinterpret_tensor(buf257, (4, 32), (32, 1), 0), buf258, buf262, buf263, buf264, reinterpret_tensor(buf265, (4, 32), (32, 1), 0), reinterpret_tensor(buf268, (4, 32), (32, 1), 0), buf269, buf273, buf274, buf276, reinterpret_tensor(buf277, (4, 32), (32, 1), 0), reinterpret_tensor(buf280, (4, 32), (32, 1), 0), buf281, buf285, buf286, buf287, reinterpret_tensor(buf288, (4, 32), (32, 1), 0), reinterpret_tensor(buf291, (4, 32), (32, 1), 0), buf292, buf296, buf297, buf298, reinterpret_tensor(buf299, (4, 32), (32, 1), 0), reinterpret_tensor(buf302, (4, 32), (32, 1), 0), buf303, buf307, buf308, buf310, reinterpret_tensor(buf311, (4, 32), (32, 1), 0), reinterpret_tensor(buf314, (4, 32), (32, 1), 0), buf315, buf319, buf320, buf325, buf326, buf327, reinterpret_tensor(buf328, (4, 32), (32, 1), 0), reinterpret_tensor(buf331, (4, 32), (32, 1), 0), buf332, buf336, buf337, buf338, reinterpret_tensor(buf339, (4, 32), (32, 1), 0), reinterpret_tensor(buf342, (4, 32), (32, 1), 0), buf343, buf347, buf348, buf350, reinterpret_tensor(buf351, (4, 32), (32, 1), 0), reinterpret_tensor(buf354, (4, 32), (32, 1), 0), buf355, buf359, buf360, buf361, reinterpret_tensor(buf362, (4, 32), (32, 1), 0), reinterpret_tensor(buf365, (4, 32), (32, 1), 0), buf366, buf370, buf371, buf372, reinterpret_tensor(buf373, (4, 32), (32, 1), 0), reinterpret_tensor(buf376, (4, 32), (32, 1), 0), buf377, buf381, buf382, buf384, reinterpret_tensor(buf385, (4, 32), (32, 1), 0), reinterpret_tensor(buf388, (4, 32), (32, 1), 0), buf389, buf393, buf394, buf395, reinterpret_tensor(buf396, (4, 32), (32, 1), 0), reinterpret_tensor(buf399, (4, 32), (32, 1), 0), buf400, buf404, buf405, buf406, reinterpret_tensor(buf407, (4, 32), (32, 1), 0), reinterpret_tensor(buf410, (4, 32), (32, 1), 0), buf411, buf415, buf416, buf418, reinterpret_tensor(buf419, (4, 32), (32, 1), 0), reinterpret_tensor(buf422, (4, 32), (32, 1), 0), buf423, buf427, buf428, buf429, reinterpret_tensor(buf430, (4, 32), (32, 1), 0), reinterpret_tensor(buf433, (4, 32), (32, 1), 0), buf434, buf438, buf439, buf440, reinterpret_tensor(buf441, (4, 32), (32, 1), 0), reinterpret_tensor(buf444, (4, 32), (32, 1), 0), buf445, buf449, buf450, buf452, reinterpret_tensor(buf453, (4, 32), (32, 1), 0), reinterpret_tensor(buf456, (4, 32), (32, 1), 0), buf457, buf461, buf462, buf467, buf468, buf469, reinterpret_tensor(buf470, (4, 32), (32, 1), 0), reinterpret_tensor(buf473, (4, 32), (32, 1), 0), buf474, buf478, buf479, buf480, reinterpret_tensor(buf481, (4, 32), (32, 1), 0), reinterpret_tensor(buf484, (4, 32), (32, 1), 0), buf485, buf489, buf490, buf492, reinterpret_tensor(buf493, (4, 32), (32, 1), 0), reinterpret_tensor(buf496, (4, 32), (32, 1), 0), buf497, buf501, buf502, buf503, reinterpret_tensor(buf504, (4, 32), (32, 1), 0), reinterpret_tensor(buf507, (4, 32), (32, 1), 0), buf508, buf512, buf513, buf514, reinterpret_tensor(buf515, (4, 32), (32, 1), 0), reinterpret_tensor(buf518, (4, 32), (32, 1), 0), buf519, buf523, buf524, buf526, reinterpret_tensor(buf527, (4, 32), (32, 1), 0), reinterpret_tensor(buf530, (4, 32), (32, 1), 0), buf531, buf535, buf536, buf537, reinterpret_tensor(buf538, (4, 32), (32, 1), 0), reinterpret_tensor(buf541, (4, 32), (32, 1), 0), buf542, buf546, buf547, buf548, reinterpret_tensor(buf549, (4, 32), (32, 1), 0), reinterpret_tensor(buf552, (4, 32), (32, 1), 0), buf553, buf557, buf558, buf560, reinterpret_tensor(buf561, (4, 32), (32, 1), 0), reinterpret_tensor(buf564, (4, 32), (32, 1), 0), buf565, buf569, buf570, buf571, reinterpret_tensor(buf572, (4, 32), (32, 1), 0), reinterpret_tensor(buf575, (4, 32), (32, 1), 0), buf576, buf580, buf581, buf582, reinterpret_tensor(buf583, (4, 32), (32, 1), 0), reinterpret_tensor(buf586, (4, 32), (32, 1), 0), buf587, buf591, buf592, buf594, buf595, buf598, buf599, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((256, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1024, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((8192, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((8192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((2048, 8192, 1, 1), (8192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((8192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((2048, 8192, 1, 1), (8192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((8192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((2048, 8192, 1, 1), (8192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((2048, 2048, 3, 3), (18432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((8192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((21843, 8192, 1, 1), (8192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((21843, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
