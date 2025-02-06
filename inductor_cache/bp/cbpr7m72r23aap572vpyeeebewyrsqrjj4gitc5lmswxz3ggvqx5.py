# AOT ID: ['57_forward']
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


# kernel path: inductor_cache/v3/cv3ojxj3bw5pa35a7wmioei64xoatl5zileue244ub64nfm6fwsz.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/pn/cpnzg3m6eaxdclupckzcamgrayghh2gnufg6lkfztdfbtsoqr2rx.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.unsqueeze]
# Source node to ATen node mapping:
#   input_1 => unsqueeze
# Graph fragment:
#   %unsqueeze : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 0), kwargs = {})
triton_poi_fused_unsqueeze_1 = async_compile.triton('triton_poi_fused_unsqueeze_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unsqueeze_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 16*y0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/42/c42ljtgumjymuxxky6dg7n74hguml5sisovmbzlj23iqwqcc6igj.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.unsqueeze]
# Source node to ATen node mapping:
#   input_2 => unsqueeze_1
# Graph fragment:
#   %unsqueeze_1 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 0), kwargs = {})
triton_poi_fused_unsqueeze_2 = async_compile.triton('triton_poi_fused_unsqueeze_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unsqueeze_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unsqueeze_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (16384 + x1 + 16*y0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eb/cebqihyubruzuehnfngkwmxsyoxm5vqaywjoyqocgxzrsoa4vc6p.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.unsqueeze]
# Source node to ATen node mapping:
#   input_3 => unsqueeze_2
# Graph fragment:
#   %unsqueeze_2 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_2, 0), kwargs = {})
triton_poi_fused_unsqueeze_3 = async_compile.triton('triton_poi_fused_unsqueeze_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unsqueeze_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unsqueeze_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (32768 + x1 + 16*y0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xc/cxcg3j5de633kd7ecr3oyi5jei5pnpqghzufyx2lizxv6yr4qoqm.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.unsqueeze]
# Source node to ATen node mapping:
#   input_4 => unsqueeze_3
# Graph fragment:
#   %unsqueeze_3 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_3, 0), kwargs = {})
triton_poi_fused_unsqueeze_4 = async_compile.triton('triton_poi_fused_unsqueeze_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unsqueeze_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (49152 + x1 + 16*y0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/46/c46lohvunymdxqfk57ycmydw243fgfzyorvbxmbieje5o7tmqgdu.py
# Topologically Sorted Source Nodes: [mid, mid_1, mid_2, input_5], Original ATen: [aten.add, aten.unsqueeze]
# Source node to ATen node mapping:
#   input_5 => unsqueeze_4
#   mid => add
#   mid_1 => add_1
#   mid_2 => add_2
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_3, %squeeze_2), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %squeeze_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %squeeze), kwargs = {})
#   %unsqueeze_4 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_3, 0), kwargs = {})
triton_poi_fused_add_unsqueeze_5 = async_compile.triton('triton_poi_fused_add_unsqueeze_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_unsqueeze_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 16*x1), tmp2, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 16*x1), tmp4, xmask & ymask)
    tl.store(out_ptr3 + (y0 + 16*x1), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/5r/c5rifcz6nwgld65uj4afo4dkgp7dbpfdgnccygcluknrsupzne6n.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.squeeze]
# Source node to ATen node mapping:
#   input_5 => squeeze_4
# Graph fragment:
#   %squeeze_4 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convolution_4, 0), kwargs = {})
triton_poi_fused_squeeze_6 = async_compile.triton('triton_poi_fused_squeeze_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_squeeze_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_squeeze_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 16*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbr6ockbdip4h26sq26nsjjrk7fb6bw7t3rtrj3b7q7okihd6ter.py
# Topologically Sorted Source Nodes: [mid, input_6], Original ATen: [aten.add, aten.unsqueeze]
# Source node to ATen node mapping:
#   input_6 => unsqueeze_5
#   mid => add
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_3, %squeeze_2), kwargs = {})
#   %unsqueeze_5 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
triton_poi_fused_add_unsqueeze_7 = async_compile.triton('triton_poi_fused_add_unsqueeze_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_unsqueeze_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_unsqueeze_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x1), tmp0, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 1024, 4, 4), (16384, 16, 4, 1))
    assert_size_stride(primals_2, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_3, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_4, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_5, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_6, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_7, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_8, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_9, (256, 256, 3, 3), (2304, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_6, buf0, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_6
        buf1 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_7, buf1, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_8, buf2, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_8
        buf3 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_9, buf3, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_9
        buf4 = empty_strided_cuda((1, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unsqueeze_1.run(primals_1, buf4, 1024, 16, grid=grid(1024, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf6 = empty_strided_cuda((1, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unsqueeze_2.run(primals_1, buf6, 1024, 16, grid=grid(1024, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf8 = empty_strided_cuda((1, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unsqueeze_3.run(primals_1, buf8, 1024, 16, grid=grid(1024, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf10 = empty_strided_cuda((1, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unsqueeze_4.run(primals_1, buf10, 1024, 16, grid=grid(1024, 16), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf12 = empty_strided_cuda((1, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        buf15 = empty_strided_cuda((256, 4, 4), (16, 4, 1), torch.float32)
        buf19 = empty_strided_cuda((256, 4, 4), (16, 4, 1), torch.float32)
        buf23 = empty_strided_cuda((256, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mid, mid_1, mid_2, input_5], Original ATen: [aten.add, aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_unsqueeze_5.run(buf11, buf9, buf7, buf5, buf12, buf15, buf19, buf23, 16, 256, grid=grid(16, 256), stream=stream0)
        del buf11
        del buf5
        del buf7
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf14 = reinterpret_tensor(buf9, (256, 4, 4), (16, 4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.squeeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_squeeze_6.run(buf13, buf14, 256, 16, grid=grid(256, 16), stream=stream0)
        buf16 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [mid, input_6], Original ATen: [aten.add, aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_unsqueeze_7.run(buf15, buf16, 256, 16, grid=grid(256, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf18 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.squeeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_squeeze_6.run(buf17, buf18, 256, 16, grid=grid(256, 16), stream=stream0)
        buf20 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [mid, mid_1, input_7], Original ATen: [aten.add, aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_unsqueeze_7.run(buf19, buf20, 256, 16, grid=grid(256, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf22 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.squeeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_squeeze_6.run(buf21, buf22, 256, 16, grid=grid(256, 16), stream=stream0)
        buf24 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [mid, mid_1, mid_2, input_8], Original ATen: [aten.add, aten.unsqueeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_unsqueeze_7.run(buf23, buf24, 256, 16, grid=grid(256, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (1, 256, 4, 4), (4096, 1, 1024, 256))
        buf26 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.squeeze]
        stream0 = get_raw_stream(0)
        triton_poi_fused_squeeze_6.run(buf25, buf26, 256, 16, grid=grid(256, 16), stream=stream0)
        del buf25
    return (buf14, buf18, buf22, buf26, primals_2, primals_3, primals_4, primals_5, buf0, buf1, buf2, buf3, buf4, buf6, buf8, buf10, buf12, buf16, buf20, buf24, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 1024, 4, 4), (16384, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
