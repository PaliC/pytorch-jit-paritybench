# AOT ID: ['14_forward']
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


# kernel path: inductor_cache/c4/cc4imz3vggpxflm4k2ligdp7sr7uxvronipdhyb2lyebvnmf6zdy.py
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
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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


# kernel path: inductor_cache/6l/c6loterk2ufuugn6npbj6jx7lmk77erdutzf3hrhnare6rlfujrt.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 26)
    y1 = yindex // 26
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 26*x2 + 234*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxgbulwbcfjdjj3wpw72twpmbdncr3v54rfflddcikump5sqrgkc.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2704
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 52)
    y1 = yindex // 52
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 52*x2 + 468*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/uz/cuza4gf5isvbjotcrnokp5aitfjz2vig7vnh44wbxey5iv53ns5a.py
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
    ynumel = 10816
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 104*x2 + 936*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zh/czhc4wia4tma3qna5fv45xu3mekyxx3lq3t42aahpiv4g7be7xw7.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 43264
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 208*x2 + 1872*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/rw/crwe5j7ttvvyizlltdagnrlilynf4o6nsbodsx7tk2une3q2s75g.py
# Topologically Sorted Source Nodes: [bn1, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   bn1 => add_1, mul_1, mul_2, sub
#   relu => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/2n/c2nq74dj3pcxwi72mvknoqe5mtnpnn4zy5x7d6gtvltsswhotqrg.py
# Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   maxpool => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 1024) % 16)
    x1 = ((xindex // 64) % 16)
    x0 = (xindex % 64)
    x5 = xindex // 1024
    x6 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-2112) + x0 + 128*x1 + 4096*x5), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2048) + x0 + 128*x1 + 4096*x5), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-1984) + x0 + 128*x1 + 4096*x5), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 4096*x5), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 4096*x5), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 4096*x5), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (1984 + x0 + 128*x1 + 4096*x5), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2048 + x0 + 128*x1 + 4096*x5), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (2112 + x0 + 128*x1 + 4096*x5), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tl.store(out_ptr0 + (x6), tmp51, None)
    tl.store(out_ptr1 + (x6), tmp76, None)
''', device_str='cuda')


# kernel path: inductor_cache/4c/c4c53hecucpriw54gi7zgljgncaaztj4dz2exfh6ca5wjmaym2rn.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   layer1_0_bn1 => add_3, mul_4, mul_5, sub_1
#   layer1_0_relu => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 104*x2 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 256*y3), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/d5/cd55rsqb67xfup5usq4msauoeaf3df3wzu5uidepiqq45jw2okus.py
# Topologically Sorted Source Nodes: [layer1_0_convs_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_convs_0 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_6, %primals_12, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 104
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 26)
    y1 = yindex // 26
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 26*x2 + 6656*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/pr/cprt36n6znuyyjmwdr7r2irgtupjlvcnrgfpyjc3siwn6yuh2ofw.py
# Topologically Sorted Source Nodes: [layer1_0_convs_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_convs_1 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_11, %primals_17, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 104
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 26)
    y1 = yindex // 26
    tmp0 = tl.load(in_ptr0 + (6656 + x2 + 256*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 26*x2 + 6656*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/df/cdfbp6pprjcmxudtmnohp357sxeuwwyfplizejjgdnar2wq3yhc2.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_2, %relu_3], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 52)
    x0 = (xindex % 256)
    x2 = xindex // 13312
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 26, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (26*x0 + 6656*x2 + (x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 52, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (26*x0 + 6656*x2 + ((-26) + x1)), tmp25, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-26) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-26) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-26) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-26) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, None)
''', device_str='cuda')


# kernel path: inductor_cache/rr/crrooajjhklm5cegbhch3p2zmsgo5eglnbwu5kjndxfdbqqtt2ew.py
# Topologically Sorted Source Nodes: [layer1_0_convs_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_convs_2 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_16, %primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 104
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 26)
    y1 = yindex // 26
    tmp0 = tl.load(in_ptr0 + (13312 + x2 + 256*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 26*x2 + 6656*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/hy/chyhxfvt7f5h377hvjektrpuzljso6nrvv2sggmnl7xjbcyiqz7j.py
# Topologically Sorted Source Nodes: [layer1_0_pool], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   layer1_0_pool => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_21, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_13 = async_compile.triton('triton_poi_fused_avg_pool2d_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x3 = xindex // 6656
    x6 = (xindex % 6656)
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (19951 + x6 + 26624*x3), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (19952 + x6 + 26624*x3), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (19953 + x6 + 26624*x3), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (19967 + x6 + 26624*x3), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (19968 + x6 + 26624*x3), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (19969 + x6 + 26624*x3), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (19983 + x6 + 26624*x3), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (19984 + x6 + 26624*x3), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (19985 + x6 + 26624*x3), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x0) + ((-1)*x1) + x0*x1 + ((17) * ((17) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (17)))*((17) * ((17) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (17))) + ((-1)*x0*((17) * ((17) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (17)))) + ((-1)*x1*((17) * ((17) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (17)))) + ((17) * ((17) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (17))) + ((17) * ((17) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (17)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6 + 26624*x3), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ud/cud3tj6srbtdlyjod2fyenvmitpn2sjftvlm5g3a566mmdx4b655.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat, %relu_4], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 79872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 256) % 78)
    x0 = (xindex % 256)
    x2 = xindex // 19968
    x3 = (xindex % 19968)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 13312*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 78, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (26*x0 + 6656*x2 + ((-52) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-52) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-52) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-52) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-52) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x3 + 26624*x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cjij2rjwhw5srcr2vtcb32ahvqmyxazzz44tidwx32wxqsmiyj4s.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_1, %avg_pool2d], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 104*x2 + 26624*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jh/cjhwgcw4nsuzaqvkhzhxeprz23xa5737ujh4uecxyhsrpccjpkt6.py
# Topologically Sorted Source Nodes: [layer1_0_bn3, layer1_0_downsample_1, add_1, layer1_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_1 => add_14
#   layer1_0_bn3 => add_11, mul_16, mul_17, sub_5
#   layer1_0_downsample_1 => add_13, mul_19, mul_20, sub_6
#   layer1_0_relu_4 => relu_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %add_13), kwargs = {})
#   %relu_5 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/yu/cyumm56tp4lu3xm32uofugalzh5c2gkb5aikbv5loe5oggybrygt.py
# Topologically Sorted Source Nodes: [layer1_1_bns_0, layer1_1_relu_1, add_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_19
#   layer1_1_bns_0 => add_18, mul_25, mul_26, sub_8
#   layer1_1_relu_1 => relu_7
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_18,), kwargs = {})
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_7, %getitem_31), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 104
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 26)
    y1 = yindex // 26
    tmp0 = tl.load(in_ptr0 + (y0 + 26*x2 + 6656*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (6656 + x2 + 256*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 256*y0 + 13312*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 26*x2 + 6656*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/r5/cr5l2xx4g4vgk3d3mtyespfnzmiqn6pw55levxvpisk5k5pdhkwx.py
# Topologically Sorted Source Nodes: [layer1_1_bns_1, layer1_1_relu_2, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_22
#   layer1_1_bns_1 => add_21, mul_28, mul_29, sub_9
#   layer1_1_relu_2 => relu_8
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_8, %getitem_36), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 104
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 26)
    y1 = yindex // 26
    tmp0 = tl.load(in_ptr0 + (y0 + 26*x2 + 6656*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (13312 + x2 + 256*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 256*y0 + 13312*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 26*x2 + 6656*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/fm/cfmkzznplpqxyp5bjd77usfauz5igvfwzf2xlezoxvq3kwwwamkj.py
# Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_6 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_4, %getitem_41], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 104)
    x1 = ((xindex // 104) % 256)
    x2 = xindex // 26624
    x3 = xindex // 104
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 78, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 52, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x1 + 256*(x0) + 13312*x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 78, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (26*x3 + ((-52) + (x0))), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr2 + ((-52) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.load(in_ptr3 + ((-52) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp28 = tl.load(in_ptr4 + ((-52) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-52) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp9, tmp11, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp4, tmp36, tmp37)
    tmp39 = tmp0 >= tmp3
    tmp40 = tl.full([1], 104, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr6 + (19968 + x1 + 256*((-78) + x0) + 26624*x2), tmp39, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.where(tmp4, tmp38, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, None)
''', device_str='cuda')


# kernel path: inductor_cache/g7/cg77c6kvgidcw5ehksqad3pigd4ilcb5vapsvewnwawcnqln36mj.py
# Topologically Sorted Source Nodes: [layer1_1_bn3, add_4, layer1_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_4 => add_27
#   layer1_1_bn3 => add_26, mul_34, mul_35, sub_11
#   layer1_1_relu_4 => relu_10
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %relu_5), kwargs = {})
#   %relu_10 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/uh/cuhtlscfb4zutotyzwxodmc4trpmgwlaf5vfze37gztkh2oll37d.py
# Topologically Sorted Source Nodes: [layer2_0_bn1, layer2_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   layer2_0_bn1 => add_42, mul_52, mul_53, sub_17
#   layer2_0_relu => relu_16
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_42,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 208*x2 + 53248*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 256*y3), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/j6/cj6mogmrobrtdy6m3srtwn6pdu2xc57yriv5uox3odjsvqrdsgst.py
# Topologically Sorted Source Nodes: [layer2_0_convs_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_convs_0 => convolution_18
# Graph fragment:
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_66, %primals_92, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_22 = async_compile.triton('triton_poi_fused_convolution_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 52)
    y1 = yindex // 52
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y0 + 53248*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 52*x2 + 13312*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/4h/c4hjkzvunx4uojihh6pys2myy5qwki72uhdklzs4diwazdzg5wyv.py
# Topologically Sorted Source Nodes: [layer2_0_convs_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_convs_1 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_71, %primals_97, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_23 = async_compile.triton('triton_poi_fused_convolution_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 52)
    y1 = yindex // 52
    tmp0 = tl.load(in_ptr0 + (13312 + x2 + 256*y0 + 53248*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 52*x2 + 13312*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/3j/c3jssdzpw4gztud67xgtv5yzhh32zdavy4jlx5qudtz57licp2sp.py
# Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_10 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_17, %relu_18], 1), kwargs = {})
triton_poi_fused_cat_24 = async_compile.triton('triton_poi_fused_cat_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 104)
    x0 = (xindex % 64)
    x2 = xindex // 6656
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (52*x0 + 3328*x2 + (x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 104, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (52*x0 + 3328*x2 + ((-52) + x1)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-52) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-52) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-52) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-52) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7b/c7buzqd3yrk7kov2mtvaqmvrdfkvfxx7xzsnwhdgprv7pj2msj7p.py
# Topologically Sorted Source Nodes: [layer2_0_convs_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_convs_2 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_76, %primals_102, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_25 = async_compile.triton('triton_poi_fused_convolution_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 52)
    y1 = yindex // 52
    tmp0 = tl.load(in_ptr0 + (26624 + x2 + 256*y0 + 53248*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 52*x2 + 13312*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ad/cad4d6wddw77kxdferppmrfbllpdis3gy2ayc3cpjpkqhwwebqwy.py
# Topologically Sorted Source Nodes: [layer2_0_pool], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   layer2_0_pool => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_81, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_26 = async_compile.triton('triton_poi_fused_avg_pool2d_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x3 = xindex // 3328
    x6 = ((xindex // 8) % 416)
    x7 = (xindex % 3328)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (39919 + 2*x0 + 32*x6 + 53248*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (39920 + 2*x0 + 32*x6 + 53248*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (39921 + 2*x0 + 32*x6 + 53248*x3), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (39935 + 2*x0 + 32*x6 + 53248*x3), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (39936 + 2*x0 + 32*x6 + 53248*x3), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (39937 + 2*x0 + 32*x6 + 53248*x3), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (39951 + 2*x0 + 32*x6 + 53248*x3), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (39952 + 2*x0 + 32*x6 + 53248*x3), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (39953 + 2*x0 + 32*x6 + 53248*x3), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((17) * ((17) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (17)))*((17) * ((17) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (17))) + ((-2)*x0*((17) * ((17) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (17)))) + ((-2)*x1*((17) * ((17) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (17)))) + 4*x0*x1 + ((17) * ((17) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (17))) + ((17) * ((17) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (17)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x7 + 13312*x3), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3r/c3rsir6jdzge4pxroedd6hpib42tewvtx654sgn7qbxwhnornesb.py
# Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_11 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_9, %relu_19], 1), kwargs = {})
triton_poi_fused_cat_27 = async_compile.triton('triton_poi_fused_cat_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 156)
    x0 = (xindex % 64)
    x2 = xindex // 9984
    x3 = (xindex % 9984)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 104, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 6656*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 156, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (52*x0 + 3328*x2 + ((-104) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-104) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-104) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-104) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-104) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x3 + 13312*x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4k/c4kki5awtgnuyfymynfwl6ggm5n22l7djgafmpghv7f3rbalvtmw.py
# Topologically Sorted Source Nodes: [cat_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_12 => cat_11
# Graph fragment:
#   %cat_11 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_10, %avg_pool2d_1], 1), kwargs = {})
triton_poi_fused_cat_28 = async_compile.triton('triton_poi_fused_cat_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 208*x2 + 13312*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/cv/ccvvgerirmmiuqzkr4yzmxs5355k5h7n336mytls6r4jk2ann64s.py
# Topologically Sorted Source Nodes: [layer2_0_bn3, layer2_0_downsample_1, add_8, layer2_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_8 => add_53
#   layer2_0_bn3 => add_50, mul_64, mul_65, sub_21
#   layer2_0_downsample_1 => add_52, mul_67, mul_68, sub_22
#   layer2_0_relu_4 => relu_20
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_50, %add_52), kwargs = {})
#   %relu_20 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/rd/crddfjjyi73jm4a6wya5n4shbaovdhutb2rxiq7univc5jhdmco4.py
# Topologically Sorted Source Nodes: [layer2_1_bn1, layer2_1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   layer2_1_bn1 => add_55, mul_70, mul_71, sub_23
#   layer2_1_relu => relu_21
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_189), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_191), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_55,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 208*x2 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 64*y3), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/nl/cnlw4hiurtywi2ier3nwyibzyq3hoov3keflrov7khfc2lisbndm.py
# Topologically Sorted Source Nodes: [layer2_1_convs_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_1_convs_0 => convolution_24
# Graph fragment:
#   %convolution_24 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_86, %primals_122, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 52)
    y1 = yindex // 52
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y0 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 52*x2 + 3328*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jc/cjczg2khgvuwe2bfc6zow7hdpde2za47an56jl7cuekku4pslhtl.py
# Topologically Sorted Source Nodes: [layer2_1_bns_0, layer2_1_relu_1, add_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_9 => add_58
#   layer2_1_bns_0 => add_57, mul_73, mul_74, sub_24
#   layer2_1_relu_1 => relu_22
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
#   %add_58 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_22, %getitem_91), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 52)
    y1 = yindex // 52
    tmp0 = tl.load(in_ptr0 + (y0 + 52*x2 + 3328*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (3328 + x2 + 64*y0 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 64*y0 + 6656*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 52*x2 + 3328*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xn/cxnbfp6l75z4aonmuuacgz7ak4z6kfmrcjus7espozhn4nqpomnr.py
# Topologically Sorted Source Nodes: [layer2_1_bns_1, layer2_1_relu_2, add_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_10 => add_61
#   layer2_1_bns_1 => add_60, mul_76, mul_77, sub_25
#   layer2_1_relu_2 => relu_23
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_23 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_60,), kwargs = {})
#   %add_61 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_23, %getitem_96), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 208
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 52)
    y1 = yindex // 52
    tmp0 = tl.load(in_ptr0 + (y0 + 52*x2 + 3328*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (6656 + x2 + 64*y0 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 64*y0 + 6656*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 52*x2 + 3328*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qc/cqcki7n7qigxtjkvrfhwzqptk2nn4sxu4y4vedx7tk43asyv4eot.py
# Topologically Sorted Source Nodes: [cat_15], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_15 => cat_14
# Graph fragment:
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_13, %getitem_101], 1), kwargs = {})
triton_poi_fused_cat_34 = async_compile.triton('triton_poi_fused_cat_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 208)
    x1 = ((xindex // 208) % 64)
    x2 = xindex // 13312
    x3 = xindex // 208
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 156, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 104, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x1 + 64*(x0) + 6656*x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 156, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (52*x3 + ((-104) + (x0))), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr2 + ((-104) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.load(in_ptr3 + ((-104) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp28 = tl.load(in_ptr4 + ((-104) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-104) + (x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp9, tmp11, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp4, tmp36, tmp37)
    tmp39 = tmp0 >= tmp3
    tmp40 = tl.full([1], 208, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr6 + (9984 + x1 + 64*((-156) + x0) + 13312*x2), tmp39, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.where(tmp4, tmp38, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, None)
''', device_str='cuda')


# kernel path: inductor_cache/gy/cgyxewsaaauufiil27zoctzweowdl3ttzoibew2n3t3kypuo5ayo.py
# Topologically Sorted Source Nodes: [layer2_1_bn3, add_11, layer2_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_11 => add_66
#   layer2_1_bn3 => add_65, mul_82, mul_83, sub_27
#   layer2_1_relu_4 => relu_25
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %relu_20), kwargs = {})
#   %relu_25 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_66,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/c5/cc5uf2njdwyl2rpbn5hazjgyoqftrz5twimzfjdzabk2uklyoxow.py
# Topologically Sorted Source Nodes: [layer3_0_bn1, layer3_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   layer3_0_bn1 => add_94, mul_115, mul_116, sub_38
#   layer3_0_relu => relu_36
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %relu_36 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_94,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 416)
    y1 = yindex // 416
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 416*x2 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 64*y3), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/pi/cpixe2ndmti7v25l5fbioiejrk7m4g5pspekubkzxdhakdevio7z.py
# Topologically Sorted Source Nodes: [layer3_0_convs_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_convs_0 => convolution_39
# Graph fragment:
#   %convolution_39 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_146, %primals_197, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_37 = async_compile.triton('triton_poi_fused_convolution_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 104*x2 + 6656*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/p4/cp4n5lkvw4gzxtksobhrognpnj4457acddhmugr5yndvvg5cxxuy.py
# Topologically Sorted Source Nodes: [layer3_0_convs_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_convs_1 => convolution_40
# Graph fragment:
#   %convolution_40 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_151, %primals_202, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (6656 + x2 + 64*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 104*x2 + 6656*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/wq/cwqrd2rp5fugoe7xmzcia3aipsg3qfhcgopkyd3w6zxj4mpc7h25.py
# Topologically Sorted Source Nodes: [cat_22], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_22 => cat_21
# Graph fragment:
#   %cat_21 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_37, %relu_38], 1), kwargs = {})
triton_poi_fused_cat_39 = async_compile.triton('triton_poi_fused_cat_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 208)
    x0 = (xindex % 16)
    x2 = xindex // 3328
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 104, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (104*x0 + 1664*x2 + (x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 208, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (104*x0 + 1664*x2 + ((-104) + x1)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-104) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-104) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-104) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-104) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cl/cclgz47yrvsov6d7tywizsgtnv53a5huzubulxas3belk2rdmrjh.py
# Topologically Sorted Source Nodes: [layer3_0_convs_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_convs_2 => convolution_41
# Graph fragment:
#   %convolution_41 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_156, %primals_207, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_40 = async_compile.triton('triton_poi_fused_convolution_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (13312 + x2 + 64*y0 + 26624*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 104*x2 + 6656*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/7x/c7x2uqqwb25hpso472wg6b6ejjfbilwqopcvqyapjwnvmmmonfme.py
# Topologically Sorted Source Nodes: [layer3_0_pool], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   layer3_0_pool => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_161, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_41 = async_compile.triton('triton_poi_fused_avg_pool2d_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x3 = xindex // 1664
    x6 = ((xindex // 4) % 416)
    x7 = (xindex % 1664)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (19959 + 2*x0 + 16*x6 + 26624*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (19960 + 2*x0 + 16*x6 + 26624*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (19961 + 2*x0 + 16*x6 + 26624*x3), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (19967 + 2*x0 + 16*x6 + 26624*x3), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (19968 + 2*x0 + 16*x6 + 26624*x3), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (19969 + 2*x0 + 16*x6 + 26624*x3), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (19975 + 2*x0 + 16*x6 + 26624*x3), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (19976 + 2*x0 + 16*x6 + 26624*x3), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (19977 + 2*x0 + 16*x6 + 26624*x3), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((9) * ((9) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (9)))*((9) * ((9) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (9))) + ((-2)*x0*((9) * ((9) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (9)))) + ((-2)*x1*((9) * ((9) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (9)))) + 4*x0*x1 + ((9) * ((9) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (9))) + ((9) * ((9) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (9)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x7 + 6656*x3), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qm/cqmglx2a27emfnfyv4sulhkeql7guq462uhj7zquoryav4pqhh4f.py
# Topologically Sorted Source Nodes: [cat_23], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_23 => cat_22
# Graph fragment:
#   %cat_22 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_21, %relu_39], 1), kwargs = {})
triton_poi_fused_cat_42 = async_compile.triton('triton_poi_fused_cat_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 312)
    x0 = (xindex % 16)
    x2 = xindex // 4992
    x3 = (xindex % 4992)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 208, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 3328*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 312, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (104*x0 + 1664*x2 + ((-208) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-208) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-208) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-208) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-208) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x3 + 6656*x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ng/cnguux3v6nmgrppwsyj6pd23yvbanl4rquhpzqypks5uv6eoruna.py
# Topologically Sorted Source Nodes: [cat_24], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_24 => cat_23
# Graph fragment:
#   %cat_23 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_22, %avg_pool2d_2], 1), kwargs = {})
triton_poi_fused_cat_43 = async_compile.triton('triton_poi_fused_cat_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_43(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 416)
    y1 = yindex // 416
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 416*x2 + 6656*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/fc/cfcxwatnkgz2ba6kacodxpb6mszpmlswdw5tegle7s752dwxrru6.py
# Topologically Sorted Source Nodes: [layer3_0_bn3, layer3_0_downsample_1, add_18, layer3_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_18 => add_105
#   layer3_0_bn3 => add_102, mul_127, mul_128, sub_42
#   layer3_0_downsample_1 => add_104, mul_130, mul_131, sub_43
#   layer3_0_relu_4 => relu_40
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_341), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_343), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_102, %add_104), kwargs = {})
#   %relu_40 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_105,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/kn/cknh7ancruw7aoq72hhxc7fvzbu777npmzzt6ek5vadfsl3s44mt.py
# Topologically Sorted Source Nodes: [layer3_1_bn1, layer3_1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   layer3_1_bn1 => add_107, mul_133, mul_134, sub_44
#   layer3_1_relu => relu_41
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_357), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_359), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 416)
    y1 = yindex // 416
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 416*x2 + 6656*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 16*y3), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/al/caloqls2tzpsbkcjqtcyquspeem44wyjx3k7jgg55nbsmpsilnby.py
# Topologically Sorted Source Nodes: [layer3_1_convs_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_1_convs_0 => convolution_45
# Graph fragment:
#   %convolution_45 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_166, %primals_227, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_46 = async_compile.triton('triton_poi_fused_convolution_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y0 + 6656*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 104*x2 + 1664*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ml/cmlde5fd56kzv5lpikl64u5l7qm7np7bivvpcr6v32hr4pqje7s2.py
# Topologically Sorted Source Nodes: [layer3_1_bns_0, layer3_1_relu_1, add_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_19 => add_110
#   layer3_1_bns_0 => add_109, mul_136, mul_137, sub_45
#   layer3_1_relu_1 => relu_42
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_109,), kwargs = {})
#   %add_110 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_42, %getitem_171), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (y0 + 104*x2 + 1664*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (1664 + x2 + 16*y0 + 6656*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 16*y0 + 3328*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 104*x2 + 1664*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/cs/ccsr7end2fqr27pnhuhsdgzqizb4lrtjkq3re3vahmuau62jq277.py
# Topologically Sorted Source Nodes: [layer3_1_bns_1, layer3_1_relu_2, add_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_20 => add_113
#   layer3_1_bns_1 => add_112, mul_139, mul_140, sub_46
#   layer3_1_relu_2 => relu_43
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_369), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_373), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_375), kwargs = {})
#   %relu_43 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_112,), kwargs = {})
#   %add_113 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_43, %getitem_176), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 416
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 104)
    y1 = yindex // 104
    tmp0 = tl.load(in_ptr0 + (y0 + 104*x2 + 1664*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (3328 + x2 + 16*y0 + 6656*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 16*y0 + 3328*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 104*x2 + 1664*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/nu/cnumwt374w7tnqyd3agqbpsqenrekmxcq7fma7hcpsie7ld7ovjq.py
# Topologically Sorted Source Nodes: [cat_27], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_27 => cat_26
# Graph fragment:
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_25, %getitem_181], 1), kwargs = {})
triton_poi_fused_cat_49 = async_compile.triton('triton_poi_fused_cat_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 416)
    x1 = ((xindex // 416) % 16)
    x2 = xindex // 6656
    x3 = xindex // 416
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 312, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 208, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x1 + 16*(x0) + 3328*x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 312, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (104*x3 + ((-208) + (x0))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr2 + ((-208) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.load(in_ptr3 + ((-208) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp28 = tl.load(in_ptr4 + ((-208) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-208) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp9, tmp11, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp4, tmp36, tmp37)
    tmp39 = tmp0 >= tmp3
    tmp40 = tl.full([1], 416, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr6 + (4992 + x1 + 16*((-312) + x0) + 6656*x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.where(tmp4, tmp38, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uk/cuklgyehx46ouergt4immpqwmw7r6o7tk7nnetvu2duhosrczfin.py
# Topologically Sorted Source Nodes: [layer3_1_bn3, add_21, layer3_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_21 => add_118
#   layer3_1_bn3 => add_117, mul_145, mul_146, sub_48
#   layer3_1_relu_4 => relu_45
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %unsqueeze_385), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_387), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_145, %unsqueeze_389), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %unsqueeze_391), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %relu_40), kwargs = {})
#   %relu_45 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_118,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/di/cdivtf2bafz7cut7zdbofmvkhmn5gv2b4w4lybubmltqeyuywxab.py
# Topologically Sorted Source Nodes: [layer4_0_bn1, layer4_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   layer4_0_bn1 => add_393, mul_463, mul_464, sub_154
#   layer4_0_relu => relu_151
# Graph fragment:
#   %sub_154 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_154, %unsqueeze_1233), kwargs = {})
#   %mul_463 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_154, %unsqueeze_1235), kwargs = {})
#   %mul_464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_463, %unsqueeze_1237), kwargs = {})
#   %add_393 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_464, %unsqueeze_1239), kwargs = {})
#   %relu_151 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_393,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 832)
    y1 = yindex // 832
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 832*x2 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 16*y3), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qx/cqxy4425h2ia4rbrbwznboxtieiipsjcqroydgphgy63br7dyirc.py
# Topologically Sorted Source Nodes: [layer4_0_convs_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_convs_0 => convolution_155
# Graph fragment:
#   %convolution_155 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_606, %primals_777, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_52 = async_compile.triton('triton_poi_fused_convolution_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_52(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y0 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 208*x2 + 3328*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/mt/cmtsuad7qjy6s7kpyyx4irae7bciuecvcd2y7jad5jjiogsiinhe.py
# Topologically Sorted Source Nodes: [layer4_0_convs_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_convs_1 => convolution_156
# Graph fragment:
#   %convolution_156 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_611, %primals_782, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_53 = async_compile.triton('triton_poi_fused_convolution_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_53(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (3328 + x2 + 16*y0 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 208*x2 + 3328*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/u3/cu3rqjmuzfkt77ppn4bnjwkfczdqoyvvcyb4l244gm7bweu4g2s7.py
# Topologically Sorted Source Nodes: [cat_91], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_91 => cat_90
# Graph fragment:
#   %cat_90 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_152, %relu_153], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 416)
    x0 = (xindex % 4)
    x2 = xindex // 1664
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 208, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (208*x0 + 832*x2 + (x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 416, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (208*x0 + 832*x2 + ((-208) + x1)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-208) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-208) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-208) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-208) + x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4e/c4egg7lwsftytfsqjp5ufccevvjqv2hlb4u2uysbjiwkpg3ecwvg.py
# Topologically Sorted Source Nodes: [layer4_0_convs_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_convs_2 => convolution_157
# Graph fragment:
#   %convolution_157 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_616, %primals_787, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_55 = async_compile.triton('triton_poi_fused_convolution_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_55(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (6656 + x2 + 16*y0 + 13312*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 208*x2 + 3328*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/sz/cszmel42a2jqadzjiiztqtr7sixo274hvovfwprt5rgnhc5eatdd.py
# Topologically Sorted Source Nodes: [layer4_0_pool], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   layer4_0_pool => avg_pool2d_3
# Graph fragment:
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_621, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_56 = async_compile.triton('triton_poi_fused_avg_pool2d_56', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x3 = xindex // 832
    x6 = ((xindex // 2) % 416)
    x7 = (xindex % 832)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (9979 + 2*x0 + 8*x6 + 13312*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (9980 + 2*x0 + 8*x6 + 13312*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (9981 + 2*x0 + 8*x6 + 13312*x3), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (9983 + 2*x0 + 8*x6 + 13312*x3), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (9984 + 2*x0 + 8*x6 + 13312*x3), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (9985 + 2*x0 + 8*x6 + 13312*x3), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (9987 + 2*x0 + 8*x6 + 13312*x3), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (9988 + 2*x0 + 8*x6 + 13312*x3), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9989 + 2*x0 + 8*x6 + 13312*x3), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5))) + ((-2)*x0*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))) + ((-2)*x1*((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))) + 4*x0*x1 + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5))) + ((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x7 + 3328*x3), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5l/c5ldeuxe4swqturja3sdle7mwqxxch7zqspu37ij345wwhts66hp.py
# Topologically Sorted Source Nodes: [cat_92], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_92 => cat_91
# Graph fragment:
#   %cat_91 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_90, %relu_154], 1), kwargs = {})
triton_poi_fused_cat_57 = async_compile.triton('triton_poi_fused_cat_57', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 624)
    x0 = (xindex % 4)
    x2 = xindex // 2496
    x3 = (xindex % 2496)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 416, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 1664*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 624, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (208*x0 + 832*x2 + ((-416) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-416) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-416) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-416) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-416) + x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x3 + 3328*x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ek/cekqpihukjfoxdq3k5fvz6nvttqquaycy3rmsejc4pucfa2zqscm.py
# Topologically Sorted Source Nodes: [cat_93], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_93 => cat_92
# Graph fragment:
#   %cat_92 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_91, %avg_pool2d_3], 1), kwargs = {})
triton_poi_fused_cat_58 = async_compile.triton('triton_poi_fused_cat_58', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_58(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 832)
    y1 = yindex // 832
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 832*x2 + 3328*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/a2/ca2tfub742jsgfbsqekavm6a67lzmjzjqgkqyo3s5m37r22vgvr5.py
# Topologically Sorted Source Nodes: [layer4_0_bn3, layer4_0_downsample_1, add_85, layer4_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_85 => add_404
#   layer4_0_bn3 => add_401, mul_475, mul_476, sub_158
#   layer4_0_downsample_1 => add_403, mul_478, mul_479, sub_159
#   layer4_0_relu_4 => relu_155
# Graph fragment:
#   %sub_158 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_158, %unsqueeze_1265), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_158, %unsqueeze_1267), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, %unsqueeze_1269), kwargs = {})
#   %add_401 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_476, %unsqueeze_1271), kwargs = {})
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_159, %unsqueeze_1273), kwargs = {})
#   %mul_478 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_478, %unsqueeze_1277), kwargs = {})
#   %add_403 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_479, %unsqueeze_1279), kwargs = {})
#   %add_404 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_401, %add_403), kwargs = {})
#   %relu_155 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_404,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/pt/cptgiusw3nl3goskm5oyibk34zmlc3weqkyvaajwgd53cm7fbnry.py
# Topologically Sorted Source Nodes: [layer4_1_bn1, layer4_1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   layer4_1_bn1 => add_406, mul_481, mul_482, sub_160
#   layer4_1_relu => relu_156
# Graph fragment:
#   %sub_160 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_160, %unsqueeze_1281), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_160, %unsqueeze_1283), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %unsqueeze_1285), kwargs = {})
#   %add_406 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_482, %unsqueeze_1287), kwargs = {})
#   %relu_156 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_406,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_60', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 832)
    y1 = yindex // 832
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 832*x2 + 3328*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 4*y3), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/np/cnphmrl3tmxouhvhpqyjnxi2qeh7lvssqykpglfv4v6bi4i6qvf2.py
# Topologically Sorted Source Nodes: [layer4_1_convs_0], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_1_convs_0 => convolution_161
# Graph fragment:
#   %convolution_161 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_626, %primals_807, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_61 = async_compile.triton('triton_poi_fused_convolution_61', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_61(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y0 + 3328*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 208*x2 + 832*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqshs4ubmzt5y577s4jokflyw47ylqtumq4hp772x2ynz32ykmz7.py
# Topologically Sorted Source Nodes: [layer4_1_bns_0, layer4_1_relu_1, add_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_86 => add_409
#   layer4_1_bns_0 => add_408, mul_484, mul_485, sub_161
#   layer4_1_relu_1 => relu_157
# Graph fragment:
#   %sub_161 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_161, %unsqueeze_1289), kwargs = {})
#   %mul_484 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_161, %unsqueeze_1291), kwargs = {})
#   %mul_485 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_484, %unsqueeze_1293), kwargs = {})
#   %add_408 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_485, %unsqueeze_1295), kwargs = {})
#   %relu_157 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_408,), kwargs = {})
#   %add_409 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_157, %getitem_631), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_62', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (y0 + 208*x2 + 832*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (832 + x2 + 4*y0 + 3328*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 4*y0 + 1664*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 208*x2 + 832*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kx/ckx4somylglthd6gee3aeip3astmgdi57uztyf665y2wbaxle7qs.py
# Topologically Sorted Source Nodes: [layer4_1_bns_1, layer4_1_relu_2, add_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_87 => add_412
#   layer4_1_bns_1 => add_411, mul_487, mul_488, sub_162
#   layer4_1_relu_2 => relu_158
# Graph fragment:
#   %sub_162 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_162, %unsqueeze_1297), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_162, %unsqueeze_1299), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_487, %unsqueeze_1301), kwargs = {})
#   %add_411 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_488, %unsqueeze_1303), kwargs = {})
#   %relu_158 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_411,), kwargs = {})
#   %add_412 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_158, %getitem_636), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 208)
    y1 = yindex // 208
    tmp0 = tl.load(in_ptr0 + (y0 + 208*x2 + 832*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (1664 + x2 + 4*y0 + 3328*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + 4*y0 + 1664*y1), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 208*x2 + 832*y1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/n3/cn3zrzcyvuvw6mfluvtj3gthjt4ajgocoj73etvmiey5kxevthjk.py
# Topologically Sorted Source Nodes: [cat_96], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_96 => cat_95
# Graph fragment:
#   %cat_95 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_94, %getitem_641], 1), kwargs = {})
triton_poi_fused_cat_64 = async_compile.triton('triton_poi_fused_cat_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 832)
    x1 = ((xindex // 832) % 4)
    x2 = xindex // 3328
    x3 = xindex // 832
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 624, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 416, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x1 + 4*(x0) + 1664*x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 624, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (208*x3 + ((-416) + (x0))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr2 + ((-416) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.load(in_ptr3 + ((-416) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp28 = tl.load(in_ptr4 + ((-416) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-416) + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp9, tmp11, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp4, tmp36, tmp37)
    tmp39 = tmp0 >= tmp3
    tmp40 = tl.full([1], 832, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr6 + (2496 + x1 + 4*((-624) + x0) + 3328*x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.where(tmp4, tmp38, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4k/c4ks6qmtplbndbazuiwcwdggsysox4qdg5qg2bn7oii2r7dlwzqg.py
# Topologically Sorted Source Nodes: [layer4_1_bn3, add_88, layer4_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_88 => add_417
#   layer4_1_bn3 => add_416, mul_493, mul_494, sub_164
#   layer4_1_relu_4 => relu_160
# Graph fragment:
#   %sub_164 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_164, %unsqueeze_1313), kwargs = {})
#   %mul_493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_164, %unsqueeze_1315), kwargs = {})
#   %mul_494 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_493, %unsqueeze_1317), kwargs = {})
#   %add_416 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_494, %unsqueeze_1319), kwargs = {})
#   %add_417 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_416, %relu_155), kwargs = {})
#   %relu_160 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_417,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/dk/cdkakag5bzj2jaotqja7hymtm3mdimlshy7wttz7h54qt5p22cnq.py
# Topologically Sorted Source Nodes: [layer4_2_bn3, add_91, layer4_2_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   add_91 => add_430
#   layer4_2_bn3 => add_429, mul_508, mul_509, sub_169
#   layer4_2_relu_4 => relu_165
# Graph fragment:
#   %sub_169 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_169, %unsqueeze_1353), kwargs = {})
#   %mul_508 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_169, %unsqueeze_1355), kwargs = {})
#   %mul_509 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_508, %unsqueeze_1357), kwargs = {})
#   %add_429 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_509, %unsqueeze_1359), kwargs = {})
#   %add_430 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_429, %relu_160), kwargs = {})
#   %relu_165 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_430,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_165, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_66 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_66', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = 0.0
    tmp21 = tmp19 <= tmp20
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/sa/csa76jwmlbe7biuwrmze4o3dm5g6vorhfxx3s4dkmqlpo3emiukm.py
# Topologically Sorted Source Nodes: [avgpool], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   avgpool => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_165, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_67 = async_compile.triton('triton_poi_fused_mean_67', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_67', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_67(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192*x1), None)
    tmp1 = tl.load(in_ptr0 + (2048 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 8192*x1), None)
    tmp5 = tl.load(in_ptr0 + (6144 + x0 + 8192*x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (104, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_8, (104, ), (1, ))
    assert_size_stride(primals_9, (104, ), (1, ))
    assert_size_stride(primals_10, (104, ), (1, ))
    assert_size_stride(primals_11, (104, ), (1, ))
    assert_size_stride(primals_12, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_13, (26, ), (1, ))
    assert_size_stride(primals_14, (26, ), (1, ))
    assert_size_stride(primals_15, (26, ), (1, ))
    assert_size_stride(primals_16, (26, ), (1, ))
    assert_size_stride(primals_17, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_18, (26, ), (1, ))
    assert_size_stride(primals_19, (26, ), (1, ))
    assert_size_stride(primals_20, (26, ), (1, ))
    assert_size_stride(primals_21, (26, ), (1, ))
    assert_size_stride(primals_22, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_23, (26, ), (1, ))
    assert_size_stride(primals_24, (26, ), (1, ))
    assert_size_stride(primals_25, (26, ), (1, ))
    assert_size_stride(primals_26, (26, ), (1, ))
    assert_size_stride(primals_27, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (104, ), (1, ))
    assert_size_stride(primals_39, (104, ), (1, ))
    assert_size_stride(primals_40, (104, ), (1, ))
    assert_size_stride(primals_41, (104, ), (1, ))
    assert_size_stride(primals_42, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_43, (26, ), (1, ))
    assert_size_stride(primals_44, (26, ), (1, ))
    assert_size_stride(primals_45, (26, ), (1, ))
    assert_size_stride(primals_46, (26, ), (1, ))
    assert_size_stride(primals_47, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_48, (26, ), (1, ))
    assert_size_stride(primals_49, (26, ), (1, ))
    assert_size_stride(primals_50, (26, ), (1, ))
    assert_size_stride(primals_51, (26, ), (1, ))
    assert_size_stride(primals_52, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_53, (26, ), (1, ))
    assert_size_stride(primals_54, (26, ), (1, ))
    assert_size_stride(primals_55, (26, ), (1, ))
    assert_size_stride(primals_56, (26, ), (1, ))
    assert_size_stride(primals_57, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (104, ), (1, ))
    assert_size_stride(primals_64, (104, ), (1, ))
    assert_size_stride(primals_65, (104, ), (1, ))
    assert_size_stride(primals_66, (104, ), (1, ))
    assert_size_stride(primals_67, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_68, (26, ), (1, ))
    assert_size_stride(primals_69, (26, ), (1, ))
    assert_size_stride(primals_70, (26, ), (1, ))
    assert_size_stride(primals_71, (26, ), (1, ))
    assert_size_stride(primals_72, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_73, (26, ), (1, ))
    assert_size_stride(primals_74, (26, ), (1, ))
    assert_size_stride(primals_75, (26, ), (1, ))
    assert_size_stride(primals_76, (26, ), (1, ))
    assert_size_stride(primals_77, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(primals_78, (26, ), (1, ))
    assert_size_stride(primals_79, (26, ), (1, ))
    assert_size_stride(primals_80, (26, ), (1, ))
    assert_size_stride(primals_81, (26, ), (1, ))
    assert_size_stride(primals_82, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (208, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_88, (208, ), (1, ))
    assert_size_stride(primals_89, (208, ), (1, ))
    assert_size_stride(primals_90, (208, ), (1, ))
    assert_size_stride(primals_91, (208, ), (1, ))
    assert_size_stride(primals_92, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_93, (52, ), (1, ))
    assert_size_stride(primals_94, (52, ), (1, ))
    assert_size_stride(primals_95, (52, ), (1, ))
    assert_size_stride(primals_96, (52, ), (1, ))
    assert_size_stride(primals_97, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_98, (52, ), (1, ))
    assert_size_stride(primals_99, (52, ), (1, ))
    assert_size_stride(primals_100, (52, ), (1, ))
    assert_size_stride(primals_101, (52, ), (1, ))
    assert_size_stride(primals_102, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_103, (52, ), (1, ))
    assert_size_stride(primals_104, (52, ), (1, ))
    assert_size_stride(primals_105, (52, ), (1, ))
    assert_size_stride(primals_106, (52, ), (1, ))
    assert_size_stride(primals_107, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (512, ), (1, ))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_112, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_118, (208, ), (1, ))
    assert_size_stride(primals_119, (208, ), (1, ))
    assert_size_stride(primals_120, (208, ), (1, ))
    assert_size_stride(primals_121, (208, ), (1, ))
    assert_size_stride(primals_122, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_123, (52, ), (1, ))
    assert_size_stride(primals_124, (52, ), (1, ))
    assert_size_stride(primals_125, (52, ), (1, ))
    assert_size_stride(primals_126, (52, ), (1, ))
    assert_size_stride(primals_127, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_128, (52, ), (1, ))
    assert_size_stride(primals_129, (52, ), (1, ))
    assert_size_stride(primals_130, (52, ), (1, ))
    assert_size_stride(primals_131, (52, ), (1, ))
    assert_size_stride(primals_132, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_133, (52, ), (1, ))
    assert_size_stride(primals_134, (52, ), (1, ))
    assert_size_stride(primals_135, (52, ), (1, ))
    assert_size_stride(primals_136, (52, ), (1, ))
    assert_size_stride(primals_137, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (512, ), (1, ))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_141, (512, ), (1, ))
    assert_size_stride(primals_142, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_143, (208, ), (1, ))
    assert_size_stride(primals_144, (208, ), (1, ))
    assert_size_stride(primals_145, (208, ), (1, ))
    assert_size_stride(primals_146, (208, ), (1, ))
    assert_size_stride(primals_147, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_148, (52, ), (1, ))
    assert_size_stride(primals_149, (52, ), (1, ))
    assert_size_stride(primals_150, (52, ), (1, ))
    assert_size_stride(primals_151, (52, ), (1, ))
    assert_size_stride(primals_152, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_153, (52, ), (1, ))
    assert_size_stride(primals_154, (52, ), (1, ))
    assert_size_stride(primals_155, (52, ), (1, ))
    assert_size_stride(primals_156, (52, ), (1, ))
    assert_size_stride(primals_157, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_158, (52, ), (1, ))
    assert_size_stride(primals_159, (52, ), (1, ))
    assert_size_stride(primals_160, (52, ), (1, ))
    assert_size_stride(primals_161, (52, ), (1, ))
    assert_size_stride(primals_162, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_168, (208, ), (1, ))
    assert_size_stride(primals_169, (208, ), (1, ))
    assert_size_stride(primals_170, (208, ), (1, ))
    assert_size_stride(primals_171, (208, ), (1, ))
    assert_size_stride(primals_172, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_173, (52, ), (1, ))
    assert_size_stride(primals_174, (52, ), (1, ))
    assert_size_stride(primals_175, (52, ), (1, ))
    assert_size_stride(primals_176, (52, ), (1, ))
    assert_size_stride(primals_177, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_178, (52, ), (1, ))
    assert_size_stride(primals_179, (52, ), (1, ))
    assert_size_stride(primals_180, (52, ), (1, ))
    assert_size_stride(primals_181, (52, ), (1, ))
    assert_size_stride(primals_182, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(primals_183, (52, ), (1, ))
    assert_size_stride(primals_184, (52, ), (1, ))
    assert_size_stride(primals_185, (52, ), (1, ))
    assert_size_stride(primals_186, (52, ), (1, ))
    assert_size_stride(primals_187, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_188, (512, ), (1, ))
    assert_size_stride(primals_189, (512, ), (1, ))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (416, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_193, (416, ), (1, ))
    assert_size_stride(primals_194, (416, ), (1, ))
    assert_size_stride(primals_195, (416, ), (1, ))
    assert_size_stride(primals_196, (416, ), (1, ))
    assert_size_stride(primals_197, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_198, (104, ), (1, ))
    assert_size_stride(primals_199, (104, ), (1, ))
    assert_size_stride(primals_200, (104, ), (1, ))
    assert_size_stride(primals_201, (104, ), (1, ))
    assert_size_stride(primals_202, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_203, (104, ), (1, ))
    assert_size_stride(primals_204, (104, ), (1, ))
    assert_size_stride(primals_205, (104, ), (1, ))
    assert_size_stride(primals_206, (104, ), (1, ))
    assert_size_stride(primals_207, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_208, (104, ), (1, ))
    assert_size_stride(primals_209, (104, ), (1, ))
    assert_size_stride(primals_210, (104, ), (1, ))
    assert_size_stride(primals_211, (104, ), (1, ))
    assert_size_stride(primals_212, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_213, (1024, ), (1, ))
    assert_size_stride(primals_214, (1024, ), (1, ))
    assert_size_stride(primals_215, (1024, ), (1, ))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (1024, ), (1, ))
    assert_size_stride(primals_220, (1024, ), (1, ))
    assert_size_stride(primals_221, (1024, ), (1, ))
    assert_size_stride(primals_222, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_223, (416, ), (1, ))
    assert_size_stride(primals_224, (416, ), (1, ))
    assert_size_stride(primals_225, (416, ), (1, ))
    assert_size_stride(primals_226, (416, ), (1, ))
    assert_size_stride(primals_227, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_228, (104, ), (1, ))
    assert_size_stride(primals_229, (104, ), (1, ))
    assert_size_stride(primals_230, (104, ), (1, ))
    assert_size_stride(primals_231, (104, ), (1, ))
    assert_size_stride(primals_232, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_233, (104, ), (1, ))
    assert_size_stride(primals_234, (104, ), (1, ))
    assert_size_stride(primals_235, (104, ), (1, ))
    assert_size_stride(primals_236, (104, ), (1, ))
    assert_size_stride(primals_237, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_238, (104, ), (1, ))
    assert_size_stride(primals_239, (104, ), (1, ))
    assert_size_stride(primals_240, (104, ), (1, ))
    assert_size_stride(primals_241, (104, ), (1, ))
    assert_size_stride(primals_242, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (1024, ), (1, ))
    assert_size_stride(primals_246, (1024, ), (1, ))
    assert_size_stride(primals_247, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_248, (416, ), (1, ))
    assert_size_stride(primals_249, (416, ), (1, ))
    assert_size_stride(primals_250, (416, ), (1, ))
    assert_size_stride(primals_251, (416, ), (1, ))
    assert_size_stride(primals_252, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_253, (104, ), (1, ))
    assert_size_stride(primals_254, (104, ), (1, ))
    assert_size_stride(primals_255, (104, ), (1, ))
    assert_size_stride(primals_256, (104, ), (1, ))
    assert_size_stride(primals_257, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_258, (104, ), (1, ))
    assert_size_stride(primals_259, (104, ), (1, ))
    assert_size_stride(primals_260, (104, ), (1, ))
    assert_size_stride(primals_261, (104, ), (1, ))
    assert_size_stride(primals_262, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_263, (104, ), (1, ))
    assert_size_stride(primals_264, (104, ), (1, ))
    assert_size_stride(primals_265, (104, ), (1, ))
    assert_size_stride(primals_266, (104, ), (1, ))
    assert_size_stride(primals_267, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_268, (1024, ), (1, ))
    assert_size_stride(primals_269, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_273, (416, ), (1, ))
    assert_size_stride(primals_274, (416, ), (1, ))
    assert_size_stride(primals_275, (416, ), (1, ))
    assert_size_stride(primals_276, (416, ), (1, ))
    assert_size_stride(primals_277, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_278, (104, ), (1, ))
    assert_size_stride(primals_279, (104, ), (1, ))
    assert_size_stride(primals_280, (104, ), (1, ))
    assert_size_stride(primals_281, (104, ), (1, ))
    assert_size_stride(primals_282, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_283, (104, ), (1, ))
    assert_size_stride(primals_284, (104, ), (1, ))
    assert_size_stride(primals_285, (104, ), (1, ))
    assert_size_stride(primals_286, (104, ), (1, ))
    assert_size_stride(primals_287, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_288, (104, ), (1, ))
    assert_size_stride(primals_289, (104, ), (1, ))
    assert_size_stride(primals_290, (104, ), (1, ))
    assert_size_stride(primals_291, (104, ), (1, ))
    assert_size_stride(primals_292, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_293, (1024, ), (1, ))
    assert_size_stride(primals_294, (1024, ), (1, ))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_297, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_298, (416, ), (1, ))
    assert_size_stride(primals_299, (416, ), (1, ))
    assert_size_stride(primals_300, (416, ), (1, ))
    assert_size_stride(primals_301, (416, ), (1, ))
    assert_size_stride(primals_302, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_303, (104, ), (1, ))
    assert_size_stride(primals_304, (104, ), (1, ))
    assert_size_stride(primals_305, (104, ), (1, ))
    assert_size_stride(primals_306, (104, ), (1, ))
    assert_size_stride(primals_307, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_308, (104, ), (1, ))
    assert_size_stride(primals_309, (104, ), (1, ))
    assert_size_stride(primals_310, (104, ), (1, ))
    assert_size_stride(primals_311, (104, ), (1, ))
    assert_size_stride(primals_312, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_313, (104, ), (1, ))
    assert_size_stride(primals_314, (104, ), (1, ))
    assert_size_stride(primals_315, (104, ), (1, ))
    assert_size_stride(primals_316, (104, ), (1, ))
    assert_size_stride(primals_317, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_319, (1024, ), (1, ))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_322, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_323, (416, ), (1, ))
    assert_size_stride(primals_324, (416, ), (1, ))
    assert_size_stride(primals_325, (416, ), (1, ))
    assert_size_stride(primals_326, (416, ), (1, ))
    assert_size_stride(primals_327, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_328, (104, ), (1, ))
    assert_size_stride(primals_329, (104, ), (1, ))
    assert_size_stride(primals_330, (104, ), (1, ))
    assert_size_stride(primals_331, (104, ), (1, ))
    assert_size_stride(primals_332, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_333, (104, ), (1, ))
    assert_size_stride(primals_334, (104, ), (1, ))
    assert_size_stride(primals_335, (104, ), (1, ))
    assert_size_stride(primals_336, (104, ), (1, ))
    assert_size_stride(primals_337, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_338, (104, ), (1, ))
    assert_size_stride(primals_339, (104, ), (1, ))
    assert_size_stride(primals_340, (104, ), (1, ))
    assert_size_stride(primals_341, (104, ), (1, ))
    assert_size_stride(primals_342, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_343, (1024, ), (1, ))
    assert_size_stride(primals_344, (1024, ), (1, ))
    assert_size_stride(primals_345, (1024, ), (1, ))
    assert_size_stride(primals_346, (1024, ), (1, ))
    assert_size_stride(primals_347, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_348, (416, ), (1, ))
    assert_size_stride(primals_349, (416, ), (1, ))
    assert_size_stride(primals_350, (416, ), (1, ))
    assert_size_stride(primals_351, (416, ), (1, ))
    assert_size_stride(primals_352, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_353, (104, ), (1, ))
    assert_size_stride(primals_354, (104, ), (1, ))
    assert_size_stride(primals_355, (104, ), (1, ))
    assert_size_stride(primals_356, (104, ), (1, ))
    assert_size_stride(primals_357, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_358, (104, ), (1, ))
    assert_size_stride(primals_359, (104, ), (1, ))
    assert_size_stride(primals_360, (104, ), (1, ))
    assert_size_stride(primals_361, (104, ), (1, ))
    assert_size_stride(primals_362, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_363, (104, ), (1, ))
    assert_size_stride(primals_364, (104, ), (1, ))
    assert_size_stride(primals_365, (104, ), (1, ))
    assert_size_stride(primals_366, (104, ), (1, ))
    assert_size_stride(primals_367, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_368, (1024, ), (1, ))
    assert_size_stride(primals_369, (1024, ), (1, ))
    assert_size_stride(primals_370, (1024, ), (1, ))
    assert_size_stride(primals_371, (1024, ), (1, ))
    assert_size_stride(primals_372, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_373, (416, ), (1, ))
    assert_size_stride(primals_374, (416, ), (1, ))
    assert_size_stride(primals_375, (416, ), (1, ))
    assert_size_stride(primals_376, (416, ), (1, ))
    assert_size_stride(primals_377, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_378, (104, ), (1, ))
    assert_size_stride(primals_379, (104, ), (1, ))
    assert_size_stride(primals_380, (104, ), (1, ))
    assert_size_stride(primals_381, (104, ), (1, ))
    assert_size_stride(primals_382, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_383, (104, ), (1, ))
    assert_size_stride(primals_384, (104, ), (1, ))
    assert_size_stride(primals_385, (104, ), (1, ))
    assert_size_stride(primals_386, (104, ), (1, ))
    assert_size_stride(primals_387, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_388, (104, ), (1, ))
    assert_size_stride(primals_389, (104, ), (1, ))
    assert_size_stride(primals_390, (104, ), (1, ))
    assert_size_stride(primals_391, (104, ), (1, ))
    assert_size_stride(primals_392, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_393, (1024, ), (1, ))
    assert_size_stride(primals_394, (1024, ), (1, ))
    assert_size_stride(primals_395, (1024, ), (1, ))
    assert_size_stride(primals_396, (1024, ), (1, ))
    assert_size_stride(primals_397, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_398, (416, ), (1, ))
    assert_size_stride(primals_399, (416, ), (1, ))
    assert_size_stride(primals_400, (416, ), (1, ))
    assert_size_stride(primals_401, (416, ), (1, ))
    assert_size_stride(primals_402, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_403, (104, ), (1, ))
    assert_size_stride(primals_404, (104, ), (1, ))
    assert_size_stride(primals_405, (104, ), (1, ))
    assert_size_stride(primals_406, (104, ), (1, ))
    assert_size_stride(primals_407, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_408, (104, ), (1, ))
    assert_size_stride(primals_409, (104, ), (1, ))
    assert_size_stride(primals_410, (104, ), (1, ))
    assert_size_stride(primals_411, (104, ), (1, ))
    assert_size_stride(primals_412, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_413, (104, ), (1, ))
    assert_size_stride(primals_414, (104, ), (1, ))
    assert_size_stride(primals_415, (104, ), (1, ))
    assert_size_stride(primals_416, (104, ), (1, ))
    assert_size_stride(primals_417, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_418, (1024, ), (1, ))
    assert_size_stride(primals_419, (1024, ), (1, ))
    assert_size_stride(primals_420, (1024, ), (1, ))
    assert_size_stride(primals_421, (1024, ), (1, ))
    assert_size_stride(primals_422, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_423, (416, ), (1, ))
    assert_size_stride(primals_424, (416, ), (1, ))
    assert_size_stride(primals_425, (416, ), (1, ))
    assert_size_stride(primals_426, (416, ), (1, ))
    assert_size_stride(primals_427, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_428, (104, ), (1, ))
    assert_size_stride(primals_429, (104, ), (1, ))
    assert_size_stride(primals_430, (104, ), (1, ))
    assert_size_stride(primals_431, (104, ), (1, ))
    assert_size_stride(primals_432, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_433, (104, ), (1, ))
    assert_size_stride(primals_434, (104, ), (1, ))
    assert_size_stride(primals_435, (104, ), (1, ))
    assert_size_stride(primals_436, (104, ), (1, ))
    assert_size_stride(primals_437, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_438, (104, ), (1, ))
    assert_size_stride(primals_439, (104, ), (1, ))
    assert_size_stride(primals_440, (104, ), (1, ))
    assert_size_stride(primals_441, (104, ), (1, ))
    assert_size_stride(primals_442, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_443, (1024, ), (1, ))
    assert_size_stride(primals_444, (1024, ), (1, ))
    assert_size_stride(primals_445, (1024, ), (1, ))
    assert_size_stride(primals_446, (1024, ), (1, ))
    assert_size_stride(primals_447, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_448, (416, ), (1, ))
    assert_size_stride(primals_449, (416, ), (1, ))
    assert_size_stride(primals_450, (416, ), (1, ))
    assert_size_stride(primals_451, (416, ), (1, ))
    assert_size_stride(primals_452, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_453, (104, ), (1, ))
    assert_size_stride(primals_454, (104, ), (1, ))
    assert_size_stride(primals_455, (104, ), (1, ))
    assert_size_stride(primals_456, (104, ), (1, ))
    assert_size_stride(primals_457, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_458, (104, ), (1, ))
    assert_size_stride(primals_459, (104, ), (1, ))
    assert_size_stride(primals_460, (104, ), (1, ))
    assert_size_stride(primals_461, (104, ), (1, ))
    assert_size_stride(primals_462, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_463, (104, ), (1, ))
    assert_size_stride(primals_464, (104, ), (1, ))
    assert_size_stride(primals_465, (104, ), (1, ))
    assert_size_stride(primals_466, (104, ), (1, ))
    assert_size_stride(primals_467, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_468, (1024, ), (1, ))
    assert_size_stride(primals_469, (1024, ), (1, ))
    assert_size_stride(primals_470, (1024, ), (1, ))
    assert_size_stride(primals_471, (1024, ), (1, ))
    assert_size_stride(primals_472, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_473, (416, ), (1, ))
    assert_size_stride(primals_474, (416, ), (1, ))
    assert_size_stride(primals_475, (416, ), (1, ))
    assert_size_stride(primals_476, (416, ), (1, ))
    assert_size_stride(primals_477, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_478, (104, ), (1, ))
    assert_size_stride(primals_479, (104, ), (1, ))
    assert_size_stride(primals_480, (104, ), (1, ))
    assert_size_stride(primals_481, (104, ), (1, ))
    assert_size_stride(primals_482, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_483, (104, ), (1, ))
    assert_size_stride(primals_484, (104, ), (1, ))
    assert_size_stride(primals_485, (104, ), (1, ))
    assert_size_stride(primals_486, (104, ), (1, ))
    assert_size_stride(primals_487, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_488, (104, ), (1, ))
    assert_size_stride(primals_489, (104, ), (1, ))
    assert_size_stride(primals_490, (104, ), (1, ))
    assert_size_stride(primals_491, (104, ), (1, ))
    assert_size_stride(primals_492, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_493, (1024, ), (1, ))
    assert_size_stride(primals_494, (1024, ), (1, ))
    assert_size_stride(primals_495, (1024, ), (1, ))
    assert_size_stride(primals_496, (1024, ), (1, ))
    assert_size_stride(primals_497, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_498, (416, ), (1, ))
    assert_size_stride(primals_499, (416, ), (1, ))
    assert_size_stride(primals_500, (416, ), (1, ))
    assert_size_stride(primals_501, (416, ), (1, ))
    assert_size_stride(primals_502, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_503, (104, ), (1, ))
    assert_size_stride(primals_504, (104, ), (1, ))
    assert_size_stride(primals_505, (104, ), (1, ))
    assert_size_stride(primals_506, (104, ), (1, ))
    assert_size_stride(primals_507, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_508, (104, ), (1, ))
    assert_size_stride(primals_509, (104, ), (1, ))
    assert_size_stride(primals_510, (104, ), (1, ))
    assert_size_stride(primals_511, (104, ), (1, ))
    assert_size_stride(primals_512, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_513, (104, ), (1, ))
    assert_size_stride(primals_514, (104, ), (1, ))
    assert_size_stride(primals_515, (104, ), (1, ))
    assert_size_stride(primals_516, (104, ), (1, ))
    assert_size_stride(primals_517, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_518, (1024, ), (1, ))
    assert_size_stride(primals_519, (1024, ), (1, ))
    assert_size_stride(primals_520, (1024, ), (1, ))
    assert_size_stride(primals_521, (1024, ), (1, ))
    assert_size_stride(primals_522, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_523, (416, ), (1, ))
    assert_size_stride(primals_524, (416, ), (1, ))
    assert_size_stride(primals_525, (416, ), (1, ))
    assert_size_stride(primals_526, (416, ), (1, ))
    assert_size_stride(primals_527, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_528, (104, ), (1, ))
    assert_size_stride(primals_529, (104, ), (1, ))
    assert_size_stride(primals_530, (104, ), (1, ))
    assert_size_stride(primals_531, (104, ), (1, ))
    assert_size_stride(primals_532, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_533, (104, ), (1, ))
    assert_size_stride(primals_534, (104, ), (1, ))
    assert_size_stride(primals_535, (104, ), (1, ))
    assert_size_stride(primals_536, (104, ), (1, ))
    assert_size_stride(primals_537, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_538, (104, ), (1, ))
    assert_size_stride(primals_539, (104, ), (1, ))
    assert_size_stride(primals_540, (104, ), (1, ))
    assert_size_stride(primals_541, (104, ), (1, ))
    assert_size_stride(primals_542, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_543, (1024, ), (1, ))
    assert_size_stride(primals_544, (1024, ), (1, ))
    assert_size_stride(primals_545, (1024, ), (1, ))
    assert_size_stride(primals_546, (1024, ), (1, ))
    assert_size_stride(primals_547, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_548, (416, ), (1, ))
    assert_size_stride(primals_549, (416, ), (1, ))
    assert_size_stride(primals_550, (416, ), (1, ))
    assert_size_stride(primals_551, (416, ), (1, ))
    assert_size_stride(primals_552, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_553, (104, ), (1, ))
    assert_size_stride(primals_554, (104, ), (1, ))
    assert_size_stride(primals_555, (104, ), (1, ))
    assert_size_stride(primals_556, (104, ), (1, ))
    assert_size_stride(primals_557, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_558, (104, ), (1, ))
    assert_size_stride(primals_559, (104, ), (1, ))
    assert_size_stride(primals_560, (104, ), (1, ))
    assert_size_stride(primals_561, (104, ), (1, ))
    assert_size_stride(primals_562, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_563, (104, ), (1, ))
    assert_size_stride(primals_564, (104, ), (1, ))
    assert_size_stride(primals_565, (104, ), (1, ))
    assert_size_stride(primals_566, (104, ), (1, ))
    assert_size_stride(primals_567, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_568, (1024, ), (1, ))
    assert_size_stride(primals_569, (1024, ), (1, ))
    assert_size_stride(primals_570, (1024, ), (1, ))
    assert_size_stride(primals_571, (1024, ), (1, ))
    assert_size_stride(primals_572, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_573, (416, ), (1, ))
    assert_size_stride(primals_574, (416, ), (1, ))
    assert_size_stride(primals_575, (416, ), (1, ))
    assert_size_stride(primals_576, (416, ), (1, ))
    assert_size_stride(primals_577, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_578, (104, ), (1, ))
    assert_size_stride(primals_579, (104, ), (1, ))
    assert_size_stride(primals_580, (104, ), (1, ))
    assert_size_stride(primals_581, (104, ), (1, ))
    assert_size_stride(primals_582, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_583, (104, ), (1, ))
    assert_size_stride(primals_584, (104, ), (1, ))
    assert_size_stride(primals_585, (104, ), (1, ))
    assert_size_stride(primals_586, (104, ), (1, ))
    assert_size_stride(primals_587, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_588, (104, ), (1, ))
    assert_size_stride(primals_589, (104, ), (1, ))
    assert_size_stride(primals_590, (104, ), (1, ))
    assert_size_stride(primals_591, (104, ), (1, ))
    assert_size_stride(primals_592, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_593, (1024, ), (1, ))
    assert_size_stride(primals_594, (1024, ), (1, ))
    assert_size_stride(primals_595, (1024, ), (1, ))
    assert_size_stride(primals_596, (1024, ), (1, ))
    assert_size_stride(primals_597, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_598, (416, ), (1, ))
    assert_size_stride(primals_599, (416, ), (1, ))
    assert_size_stride(primals_600, (416, ), (1, ))
    assert_size_stride(primals_601, (416, ), (1, ))
    assert_size_stride(primals_602, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_603, (104, ), (1, ))
    assert_size_stride(primals_604, (104, ), (1, ))
    assert_size_stride(primals_605, (104, ), (1, ))
    assert_size_stride(primals_606, (104, ), (1, ))
    assert_size_stride(primals_607, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_608, (104, ), (1, ))
    assert_size_stride(primals_609, (104, ), (1, ))
    assert_size_stride(primals_610, (104, ), (1, ))
    assert_size_stride(primals_611, (104, ), (1, ))
    assert_size_stride(primals_612, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_613, (104, ), (1, ))
    assert_size_stride(primals_614, (104, ), (1, ))
    assert_size_stride(primals_615, (104, ), (1, ))
    assert_size_stride(primals_616, (104, ), (1, ))
    assert_size_stride(primals_617, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_618, (1024, ), (1, ))
    assert_size_stride(primals_619, (1024, ), (1, ))
    assert_size_stride(primals_620, (1024, ), (1, ))
    assert_size_stride(primals_621, (1024, ), (1, ))
    assert_size_stride(primals_622, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_623, (416, ), (1, ))
    assert_size_stride(primals_624, (416, ), (1, ))
    assert_size_stride(primals_625, (416, ), (1, ))
    assert_size_stride(primals_626, (416, ), (1, ))
    assert_size_stride(primals_627, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_628, (104, ), (1, ))
    assert_size_stride(primals_629, (104, ), (1, ))
    assert_size_stride(primals_630, (104, ), (1, ))
    assert_size_stride(primals_631, (104, ), (1, ))
    assert_size_stride(primals_632, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_633, (104, ), (1, ))
    assert_size_stride(primals_634, (104, ), (1, ))
    assert_size_stride(primals_635, (104, ), (1, ))
    assert_size_stride(primals_636, (104, ), (1, ))
    assert_size_stride(primals_637, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_638, (104, ), (1, ))
    assert_size_stride(primals_639, (104, ), (1, ))
    assert_size_stride(primals_640, (104, ), (1, ))
    assert_size_stride(primals_641, (104, ), (1, ))
    assert_size_stride(primals_642, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_643, (1024, ), (1, ))
    assert_size_stride(primals_644, (1024, ), (1, ))
    assert_size_stride(primals_645, (1024, ), (1, ))
    assert_size_stride(primals_646, (1024, ), (1, ))
    assert_size_stride(primals_647, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_648, (416, ), (1, ))
    assert_size_stride(primals_649, (416, ), (1, ))
    assert_size_stride(primals_650, (416, ), (1, ))
    assert_size_stride(primals_651, (416, ), (1, ))
    assert_size_stride(primals_652, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_653, (104, ), (1, ))
    assert_size_stride(primals_654, (104, ), (1, ))
    assert_size_stride(primals_655, (104, ), (1, ))
    assert_size_stride(primals_656, (104, ), (1, ))
    assert_size_stride(primals_657, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_658, (104, ), (1, ))
    assert_size_stride(primals_659, (104, ), (1, ))
    assert_size_stride(primals_660, (104, ), (1, ))
    assert_size_stride(primals_661, (104, ), (1, ))
    assert_size_stride(primals_662, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_663, (104, ), (1, ))
    assert_size_stride(primals_664, (104, ), (1, ))
    assert_size_stride(primals_665, (104, ), (1, ))
    assert_size_stride(primals_666, (104, ), (1, ))
    assert_size_stride(primals_667, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_668, (1024, ), (1, ))
    assert_size_stride(primals_669, (1024, ), (1, ))
    assert_size_stride(primals_670, (1024, ), (1, ))
    assert_size_stride(primals_671, (1024, ), (1, ))
    assert_size_stride(primals_672, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_673, (416, ), (1, ))
    assert_size_stride(primals_674, (416, ), (1, ))
    assert_size_stride(primals_675, (416, ), (1, ))
    assert_size_stride(primals_676, (416, ), (1, ))
    assert_size_stride(primals_677, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_678, (104, ), (1, ))
    assert_size_stride(primals_679, (104, ), (1, ))
    assert_size_stride(primals_680, (104, ), (1, ))
    assert_size_stride(primals_681, (104, ), (1, ))
    assert_size_stride(primals_682, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_683, (104, ), (1, ))
    assert_size_stride(primals_684, (104, ), (1, ))
    assert_size_stride(primals_685, (104, ), (1, ))
    assert_size_stride(primals_686, (104, ), (1, ))
    assert_size_stride(primals_687, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_688, (104, ), (1, ))
    assert_size_stride(primals_689, (104, ), (1, ))
    assert_size_stride(primals_690, (104, ), (1, ))
    assert_size_stride(primals_691, (104, ), (1, ))
    assert_size_stride(primals_692, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_693, (1024, ), (1, ))
    assert_size_stride(primals_694, (1024, ), (1, ))
    assert_size_stride(primals_695, (1024, ), (1, ))
    assert_size_stride(primals_696, (1024, ), (1, ))
    assert_size_stride(primals_697, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_698, (416, ), (1, ))
    assert_size_stride(primals_699, (416, ), (1, ))
    assert_size_stride(primals_700, (416, ), (1, ))
    assert_size_stride(primals_701, (416, ), (1, ))
    assert_size_stride(primals_702, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_703, (104, ), (1, ))
    assert_size_stride(primals_704, (104, ), (1, ))
    assert_size_stride(primals_705, (104, ), (1, ))
    assert_size_stride(primals_706, (104, ), (1, ))
    assert_size_stride(primals_707, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_708, (104, ), (1, ))
    assert_size_stride(primals_709, (104, ), (1, ))
    assert_size_stride(primals_710, (104, ), (1, ))
    assert_size_stride(primals_711, (104, ), (1, ))
    assert_size_stride(primals_712, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_713, (104, ), (1, ))
    assert_size_stride(primals_714, (104, ), (1, ))
    assert_size_stride(primals_715, (104, ), (1, ))
    assert_size_stride(primals_716, (104, ), (1, ))
    assert_size_stride(primals_717, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_718, (1024, ), (1, ))
    assert_size_stride(primals_719, (1024, ), (1, ))
    assert_size_stride(primals_720, (1024, ), (1, ))
    assert_size_stride(primals_721, (1024, ), (1, ))
    assert_size_stride(primals_722, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_723, (416, ), (1, ))
    assert_size_stride(primals_724, (416, ), (1, ))
    assert_size_stride(primals_725, (416, ), (1, ))
    assert_size_stride(primals_726, (416, ), (1, ))
    assert_size_stride(primals_727, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_728, (104, ), (1, ))
    assert_size_stride(primals_729, (104, ), (1, ))
    assert_size_stride(primals_730, (104, ), (1, ))
    assert_size_stride(primals_731, (104, ), (1, ))
    assert_size_stride(primals_732, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_733, (104, ), (1, ))
    assert_size_stride(primals_734, (104, ), (1, ))
    assert_size_stride(primals_735, (104, ), (1, ))
    assert_size_stride(primals_736, (104, ), (1, ))
    assert_size_stride(primals_737, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_738, (104, ), (1, ))
    assert_size_stride(primals_739, (104, ), (1, ))
    assert_size_stride(primals_740, (104, ), (1, ))
    assert_size_stride(primals_741, (104, ), (1, ))
    assert_size_stride(primals_742, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_743, (1024, ), (1, ))
    assert_size_stride(primals_744, (1024, ), (1, ))
    assert_size_stride(primals_745, (1024, ), (1, ))
    assert_size_stride(primals_746, (1024, ), (1, ))
    assert_size_stride(primals_747, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_748, (416, ), (1, ))
    assert_size_stride(primals_749, (416, ), (1, ))
    assert_size_stride(primals_750, (416, ), (1, ))
    assert_size_stride(primals_751, (416, ), (1, ))
    assert_size_stride(primals_752, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_753, (104, ), (1, ))
    assert_size_stride(primals_754, (104, ), (1, ))
    assert_size_stride(primals_755, (104, ), (1, ))
    assert_size_stride(primals_756, (104, ), (1, ))
    assert_size_stride(primals_757, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_758, (104, ), (1, ))
    assert_size_stride(primals_759, (104, ), (1, ))
    assert_size_stride(primals_760, (104, ), (1, ))
    assert_size_stride(primals_761, (104, ), (1, ))
    assert_size_stride(primals_762, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(primals_763, (104, ), (1, ))
    assert_size_stride(primals_764, (104, ), (1, ))
    assert_size_stride(primals_765, (104, ), (1, ))
    assert_size_stride(primals_766, (104, ), (1, ))
    assert_size_stride(primals_767, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_768, (1024, ), (1, ))
    assert_size_stride(primals_769, (1024, ), (1, ))
    assert_size_stride(primals_770, (1024, ), (1, ))
    assert_size_stride(primals_771, (1024, ), (1, ))
    assert_size_stride(primals_772, (832, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_773, (832, ), (1, ))
    assert_size_stride(primals_774, (832, ), (1, ))
    assert_size_stride(primals_775, (832, ), (1, ))
    assert_size_stride(primals_776, (832, ), (1, ))
    assert_size_stride(primals_777, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_778, (208, ), (1, ))
    assert_size_stride(primals_779, (208, ), (1, ))
    assert_size_stride(primals_780, (208, ), (1, ))
    assert_size_stride(primals_781, (208, ), (1, ))
    assert_size_stride(primals_782, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_783, (208, ), (1, ))
    assert_size_stride(primals_784, (208, ), (1, ))
    assert_size_stride(primals_785, (208, ), (1, ))
    assert_size_stride(primals_786, (208, ), (1, ))
    assert_size_stride(primals_787, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_788, (208, ), (1, ))
    assert_size_stride(primals_789, (208, ), (1, ))
    assert_size_stride(primals_790, (208, ), (1, ))
    assert_size_stride(primals_791, (208, ), (1, ))
    assert_size_stride(primals_792, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_793, (2048, ), (1, ))
    assert_size_stride(primals_794, (2048, ), (1, ))
    assert_size_stride(primals_795, (2048, ), (1, ))
    assert_size_stride(primals_796, (2048, ), (1, ))
    assert_size_stride(primals_797, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_798, (2048, ), (1, ))
    assert_size_stride(primals_799, (2048, ), (1, ))
    assert_size_stride(primals_800, (2048, ), (1, ))
    assert_size_stride(primals_801, (2048, ), (1, ))
    assert_size_stride(primals_802, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_803, (832, ), (1, ))
    assert_size_stride(primals_804, (832, ), (1, ))
    assert_size_stride(primals_805, (832, ), (1, ))
    assert_size_stride(primals_806, (832, ), (1, ))
    assert_size_stride(primals_807, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_808, (208, ), (1, ))
    assert_size_stride(primals_809, (208, ), (1, ))
    assert_size_stride(primals_810, (208, ), (1, ))
    assert_size_stride(primals_811, (208, ), (1, ))
    assert_size_stride(primals_812, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_813, (208, ), (1, ))
    assert_size_stride(primals_814, (208, ), (1, ))
    assert_size_stride(primals_815, (208, ), (1, ))
    assert_size_stride(primals_816, (208, ), (1, ))
    assert_size_stride(primals_817, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_818, (208, ), (1, ))
    assert_size_stride(primals_819, (208, ), (1, ))
    assert_size_stride(primals_820, (208, ), (1, ))
    assert_size_stride(primals_821, (208, ), (1, ))
    assert_size_stride(primals_822, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_823, (2048, ), (1, ))
    assert_size_stride(primals_824, (2048, ), (1, ))
    assert_size_stride(primals_825, (2048, ), (1, ))
    assert_size_stride(primals_826, (2048, ), (1, ))
    assert_size_stride(primals_827, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_828, (832, ), (1, ))
    assert_size_stride(primals_829, (832, ), (1, ))
    assert_size_stride(primals_830, (832, ), (1, ))
    assert_size_stride(primals_831, (832, ), (1, ))
    assert_size_stride(primals_832, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_833, (208, ), (1, ))
    assert_size_stride(primals_834, (208, ), (1, ))
    assert_size_stride(primals_835, (208, ), (1, ))
    assert_size_stride(primals_836, (208, ), (1, ))
    assert_size_stride(primals_837, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_838, (208, ), (1, ))
    assert_size_stride(primals_839, (208, ), (1, ))
    assert_size_stride(primals_840, (208, ), (1, ))
    assert_size_stride(primals_841, (208, ), (1, ))
    assert_size_stride(primals_842, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(primals_843, (208, ), (1, ))
    assert_size_stride(primals_844, (208, ), (1, ))
    assert_size_stride(primals_845, (208, ), (1, ))
    assert_size_stride(primals_846, (208, ), (1, ))
    assert_size_stride(primals_847, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_848, (2048, ), (1, ))
    assert_size_stride(primals_849, (2048, ), (1, ))
    assert_size_stride(primals_850, (2048, ), (1, ))
    assert_size_stride(primals_851, (2048, ), (1, ))
    assert_size_stride(primals_852, (1000, 2048), (2048, 1))
    assert_size_stride(primals_853, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 49, grid=grid(192, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_12, buf2, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_12
        buf3 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_17, buf3, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_17
        buf4 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_22, buf4, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_22
        buf5 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_42, buf5, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_42
        buf6 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_47, buf6, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_47
        buf7 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_52, buf7, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_52
        buf8 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_67, buf8, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_67
        buf9 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_72, buf9, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_72
        buf10 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_77, buf10, 676, 9, grid=grid(676, 9), stream=stream0)
        del primals_77
        buf11 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_92, buf11, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_92
        buf12 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_97, buf12, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_97
        buf13 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_102, buf13, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_102
        buf14 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_122, buf14, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_122
        buf15 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_127, buf15, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_127
        buf16 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_132, buf16, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_132
        buf17 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_147, buf17, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_147
        buf18 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_152, buf18, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_152
        buf19 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_157, buf19, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_157
        buf20 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_172, buf20, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_172
        buf21 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_177, buf21, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_177
        buf22 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_182, buf22, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del primals_182
        buf23 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_197, buf23, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_197
        buf24 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_202, buf24, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_202
        buf25 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_207, buf25, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_207
        buf26 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_227, buf26, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_227
        buf27 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_232, buf27, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_232
        buf28 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_237, buf28, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_237
        buf29 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_252, buf29, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_252
        buf30 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_257, buf30, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_257
        buf31 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_262, buf31, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_262
        buf32 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_277, buf32, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_277
        buf33 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_282, buf33, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_282
        buf34 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_287, buf34, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_287
        buf35 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_302, buf35, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_302
        buf36 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_307, buf36, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_307
        buf37 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_312, buf37, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_312
        buf38 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_327, buf38, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_327
        buf39 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_332, buf39, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_332
        buf40 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_337, buf40, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_337
        buf41 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_352, buf41, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_352
        buf42 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_357, buf42, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_357
        buf43 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_362, buf43, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_362
        buf44 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_377, buf44, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_377
        buf45 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_382, buf45, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_382
        buf46 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_387, buf46, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_387
        buf47 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_402, buf47, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_402
        buf48 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_407, buf48, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_407
        buf49 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_412, buf49, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_412
        buf50 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_427, buf50, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_427
        buf51 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_432, buf51, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_432
        buf52 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_437, buf52, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_437
        buf53 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_452, buf53, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_452
        buf54 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_457, buf54, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_457
        buf55 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_462, buf55, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_462
        buf56 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_477, buf56, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_477
        buf57 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_482, buf57, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_482
        buf58 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_487, buf58, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_487
        buf59 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_502, buf59, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_502
        buf60 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_507, buf60, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_507
        buf61 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_512, buf61, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_512
        buf62 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_527, buf62, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_527
        buf63 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_532, buf63, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_532
        buf64 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_537, buf64, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_537
        buf65 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_552, buf65, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_552
        buf66 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_557, buf66, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_557
        buf67 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_562, buf67, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_562
        buf68 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_577, buf68, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_577
        buf69 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_582, buf69, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_582
        buf70 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_587, buf70, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_587
        buf71 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_602, buf71, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_602
        buf72 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_607, buf72, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_607
        buf73 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_612, buf73, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_612
        buf74 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_627, buf74, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_627
        buf75 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_632, buf75, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_632
        buf76 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_637, buf76, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_637
        buf77 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_652, buf77, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_652
        buf78 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_657, buf78, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_657
        buf79 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_662, buf79, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_662
        buf80 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_677, buf80, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_677
        buf81 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_682, buf81, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_682
        buf82 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_687, buf82, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_687
        buf83 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_702, buf83, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_702
        buf84 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_707, buf84, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_707
        buf85 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_712, buf85, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_712
        buf86 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_727, buf86, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_727
        buf87 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_732, buf87, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_732
        buf88 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_737, buf88, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_737
        buf89 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_752, buf89, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_752
        buf90 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_757, buf90, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_757
        buf91 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_762, buf91, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del primals_762
        buf92 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_777, buf92, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_777
        buf93 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_782, buf93, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_782
        buf94 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_787, buf94, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_787
        buf95 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_807, buf95, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_807
        buf96 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_812, buf96, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_812
        buf97 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_817, buf97, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_817
        buf98 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_832, buf98, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_832
        buf99 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_837, buf99, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_837
        buf100 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_842, buf100, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del primals_842
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf102 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [bn1, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf101, primals_3, primals_4, primals_5, primals_6, buf102, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        buf103 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf104 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.int8)
        # Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_7.run(buf102, buf103, buf104, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf103, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 104, 16, 16), (26624, 1, 1664, 104))
        buf106 = empty_strided_cuda((4, 104, 16, 16), (26624, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf105, primals_8, primals_9, primals_10, primals_11, buf106, 416, 256, grid=grid(416, 256), stream=stream0)
        buf107 = empty_strided_cuda((4, 26, 16, 16), (6656, 1, 416, 26), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf106, buf107, 104, 256, grid=grid(104, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_convs_0], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf109 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_convs_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(buf106, buf109, 104, 256, grid=grid(104, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_convs_1], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf111 = empty_strided_cuda((4, 52, 16, 16), (13312, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf108, primals_13, primals_14, primals_15, primals_16, buf110, primals_18, primals_19, primals_20, primals_21, buf111, 53248, grid=grid(53248), stream=stream0)
        buf112 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_convs_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf106, buf112, 104, 256, grid=grid(104, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_convs_2], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf116 = empty_strided_cuda((4, 104, 16, 16), (26624, 256, 16, 1), torch.float32)
        buf114 = reinterpret_tensor(buf116, (4, 26, 16, 16), (26624, 256, 16, 1), 19968)  # alias
        # Topologically Sorted Source Nodes: [layer1_0_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_13.run(buf106, buf114, 26624, grid=grid(26624), stream=stream0)
        buf115 = reinterpret_tensor(buf116, (4, 78, 16, 16), (26624, 256, 16, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf111, buf113, primals_23, primals_24, primals_25, primals_26, buf115, 79872, grid=grid(79872), stream=stream0)
        buf117 = empty_strided_cuda((4, 104, 16, 16), (26624, 1, 1664, 104), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf116, buf117, 416, 256, grid=grid(416, 256), stream=stream0)
        del buf114
        del buf115
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 256, 16, 16), (65536, 1, 4096, 256))
        # Topologically Sorted Source Nodes: [layer1_0_downsample_0], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf103, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf120 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_bn3, layer1_0_downsample_1, add_1, layer1_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf121, buf118, primals_28, primals_29, primals_30, primals_31, buf119, primals_33, primals_34, primals_35, primals_36, 262144, grid=grid(262144), stream=stream0)
        del primals_31
        del primals_36
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 104, 16, 16), (26624, 1, 1664, 104))
        buf123 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_bn1, layer1_1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf122, primals_38, primals_39, primals_40, primals_41, buf123, 416, 256, grid=grid(416, 256), stream=stream0)
        buf124 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf123, buf124, 104, 256, grid=grid(104, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_convs_0], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf130 = buf111; del buf111  # reuse
        buf126 = reinterpret_tensor(buf130, (4, 26, 16, 16), (13312, 256, 16, 1), 0)  # alias
        buf127 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_bns_0, layer1_1_relu_1, add_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf125, primals_43, primals_44, primals_45, primals_46, buf123, buf126, buf127, 104, 256, grid=grid(104, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_convs_1], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf129 = reinterpret_tensor(buf130, (4, 26, 16, 16), (13312, 256, 16, 1), 6656)  # alias
        buf131 = empty_strided_cuda((4, 26, 16, 16), (6656, 1, 416, 26), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_1_bns_1, layer1_1_relu_2, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf128, primals_48, primals_49, primals_50, primals_51, buf123, buf129, buf131, 104, 256, grid=grid(104, 256), stream=stream0)
        del buf126
        del buf129
        # Topologically Sorted Source Nodes: [layer1_1_convs_2], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf133 = empty_strided_cuda((4, 104, 16, 16), (26624, 1, 1664, 104), torch.float32)
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf130, buf132, primals_53, primals_54, primals_55, primals_56, buf123, buf133, 106496, grid=grid(106496), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf135 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_1_bn3, add_4, layer1_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf134, primals_58, primals_59, primals_60, primals_61, buf121, buf135, 262144, grid=grid(262144), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 104, 16, 16), (26624, 1, 1664, 104))
        buf137 = empty_strided_cuda((4, 104, 16, 16), (26624, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_2_bn1, layer1_2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf136, primals_63, primals_64, primals_65, primals_66, buf137, 416, 256, grid=grid(416, 256), stream=stream0)
        buf138 = empty_strided_cuda((4, 26, 16, 16), (6656, 1, 416, 26), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_2_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf137, buf138, 104, 256, grid=grid(104, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_convs_0], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf144 = buf130; del buf130  # reuse
        buf140 = reinterpret_tensor(buf144, (4, 26, 16, 16), (13312, 256, 16, 1), 0)  # alias
        buf141 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_bns_0, layer1_2_relu_1, add_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf139, primals_68, primals_69, primals_70, primals_71, buf137, buf140, buf141, 104, 256, grid=grid(104, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_convs_1], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf143 = reinterpret_tensor(buf144, (4, 26, 16, 16), (13312, 256, 16, 1), 6656)  # alias
        buf145 = empty_strided_cuda((4, 26, 16, 16), (6656, 1, 416, 26), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_2_bns_1, layer1_2_relu_2, add_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf142, primals_73, primals_74, primals_75, primals_76, buf137, buf143, buf145, 104, 256, grid=grid(104, 256), stream=stream0)
        del buf140
        del buf143
        # Topologically Sorted Source Nodes: [layer1_2_convs_2], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 26, 16, 16), (6656, 1, 416, 26))
        buf147 = empty_strided_cuda((4, 104, 16, 16), (26624, 1, 1664, 104), torch.float32)
        # Topologically Sorted Source Nodes: [cat_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf144, buf146, primals_78, primals_79, primals_80, primals_81, buf137, buf147, 106496, grid=grid(106496), stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf149 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_2_bn3, add_7, layer1_2_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf148, primals_83, primals_84, primals_85, primals_86, buf135, buf149, 262144, grid=grid(262144), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 208, 16, 16), (53248, 1, 3328, 208))
        buf151 = empty_strided_cuda((4, 208, 16, 16), (53248, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn1, layer2_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf150, primals_88, primals_89, primals_90, primals_91, buf151, 832, 256, grid=grid(832, 256), stream=stream0)
        buf152 = reinterpret_tensor(buf144, (4, 52, 16, 16), (13312, 1, 832, 52), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf151, buf152, 208, 256, grid=grid(208, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_convs_0], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, buf11, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf154 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_convs_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_23.run(buf151, buf154, 208, 256, grid=grid(208, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_convs_1], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf156 = empty_strided_cuda((4, 104, 8, 8), (6656, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_24.run(buf153, primals_93, primals_94, primals_95, primals_96, buf155, primals_98, primals_99, primals_100, primals_101, buf156, 26624, grid=grid(26624), stream=stream0)
        buf157 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_convs_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_25.run(buf151, buf157, 208, 256, grid=grid(208, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_convs_2], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, buf13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf161 = reinterpret_tensor(buf157, (4, 208, 8, 8), (13312, 64, 8, 1), 0); del buf157  # reuse
        buf159 = reinterpret_tensor(buf161, (4, 52, 8, 8), (13312, 64, 8, 1), 9984)  # alias
        # Topologically Sorted Source Nodes: [layer2_0_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_26.run(buf151, buf159, 13312, grid=grid(13312), stream=stream0)
        buf160 = reinterpret_tensor(buf161, (4, 156, 8, 8), (13312, 64, 8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_27.run(buf156, buf158, primals_103, primals_104, primals_105, primals_106, buf160, 39936, grid=grid(39936), stream=stream0)
        buf162 = empty_strided_cuda((4, 208, 8, 8), (13312, 1, 1664, 208), torch.float32)
        # Topologically Sorted Source Nodes: [cat_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_28.run(buf161, buf162, 832, 64, grid=grid(832, 64), stream=stream0)
        del buf159
        del buf160
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 512, 8, 8), (32768, 1, 4096, 512))
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf149, primals_112, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf165 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_bn3, layer2_0_downsample_1, add_8, layer2_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_29.run(buf166, buf163, primals_108, primals_109, primals_110, primals_111, buf164, primals_113, primals_114, primals_115, primals_116, 131072, grid=grid(131072), stream=stream0)
        del primals_111
        del primals_116
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 208, 8, 8), (13312, 1, 1664, 208))
        buf168 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_bn1, layer2_1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf167, primals_118, primals_119, primals_120, primals_121, buf168, 832, 64, grid=grid(832, 64), stream=stream0)
        buf169 = empty_strided_cuda((4, 52, 8, 8), (3328, 1, 416, 52), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_1_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf168, buf169, 208, 64, grid=grid(208, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_convs_0], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf175 = buf156; del buf156  # reuse
        buf171 = reinterpret_tensor(buf175, (4, 52, 8, 8), (6656, 64, 8, 1), 0)  # alias
        buf172 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_bns_0, layer2_1_relu_1, add_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32.run(buf170, primals_123, primals_124, primals_125, primals_126, buf168, buf171, buf172, 208, 64, grid=grid(208, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_convs_1], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf174 = reinterpret_tensor(buf175, (4, 52, 8, 8), (6656, 64, 8, 1), 3328)  # alias
        buf176 = empty_strided_cuda((4, 52, 8, 8), (3328, 1, 416, 52), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_1_bns_1, layer2_1_relu_2, add_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf173, primals_128, primals_129, primals_130, primals_131, buf168, buf174, buf176, 208, 64, grid=grid(208, 64), stream=stream0)
        del buf171
        del buf174
        # Topologically Sorted Source Nodes: [layer2_1_convs_2], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf178 = empty_strided_cuda((4, 208, 8, 8), (13312, 1, 1664, 208), torch.float32)
        # Topologically Sorted Source Nodes: [cat_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_34.run(buf175, buf177, primals_133, primals_134, primals_135, primals_136, buf168, buf178, 53248, grid=grid(53248), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf180 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_1_bn3, add_11, layer2_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_35.run(buf179, primals_138, primals_139, primals_140, primals_141, buf166, buf180, 131072, grid=grid(131072), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 208, 8, 8), (13312, 1, 1664, 208))
        buf182 = empty_strided_cuda((4, 208, 8, 8), (13312, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_2_bn1, layer2_2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf181, primals_143, primals_144, primals_145, primals_146, buf182, 832, 64, grid=grid(832, 64), stream=stream0)
        buf183 = empty_strided_cuda((4, 52, 8, 8), (3328, 1, 416, 52), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_2_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf182, buf183, 208, 64, grid=grid(208, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_convs_0], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf189 = buf175; del buf175  # reuse
        buf185 = reinterpret_tensor(buf189, (4, 52, 8, 8), (6656, 64, 8, 1), 0)  # alias
        buf186 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_bns_0, layer2_2_relu_1, add_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32.run(buf184, primals_148, primals_149, primals_150, primals_151, buf182, buf185, buf186, 208, 64, grid=grid(208, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_convs_1], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf188 = reinterpret_tensor(buf189, (4, 52, 8, 8), (6656, 64, 8, 1), 3328)  # alias
        buf190 = empty_strided_cuda((4, 52, 8, 8), (3328, 1, 416, 52), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_2_bns_1, layer2_2_relu_2, add_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf187, primals_153, primals_154, primals_155, primals_156, buf182, buf188, buf190, 208, 64, grid=grid(208, 64), stream=stream0)
        del buf185
        del buf188
        # Topologically Sorted Source Nodes: [layer2_2_convs_2], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf192 = empty_strided_cuda((4, 208, 8, 8), (13312, 1, 1664, 208), torch.float32)
        # Topologically Sorted Source Nodes: [cat_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_34.run(buf189, buf191, primals_158, primals_159, primals_160, primals_161, buf182, buf192, 53248, grid=grid(53248), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf194 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_2_bn3, add_14, layer2_2_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_35.run(buf193, primals_163, primals_164, primals_165, primals_166, buf180, buf194, 131072, grid=grid(131072), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 208, 8, 8), (13312, 1, 1664, 208))
        buf196 = empty_strided_cuda((4, 208, 8, 8), (13312, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_3_bn1, layer2_3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf195, primals_168, primals_169, primals_170, primals_171, buf196, 832, 64, grid=grid(832, 64), stream=stream0)
        buf197 = empty_strided_cuda((4, 52, 8, 8), (3328, 1, 416, 52), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_3_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf196, buf197, 208, 64, grid=grid(208, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_convs_0], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf203 = buf189; del buf189  # reuse
        buf199 = reinterpret_tensor(buf203, (4, 52, 8, 8), (6656, 64, 8, 1), 0)  # alias
        buf200 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_bns_0, layer2_3_relu_1, add_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32.run(buf198, primals_173, primals_174, primals_175, primals_176, buf196, buf199, buf200, 208, 64, grid=grid(208, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_convs_1], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf202 = reinterpret_tensor(buf203, (4, 52, 8, 8), (6656, 64, 8, 1), 3328)  # alias
        buf204 = empty_strided_cuda((4, 52, 8, 8), (3328, 1, 416, 52), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_3_bns_1, layer2_3_relu_2, add_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf201, primals_178, primals_179, primals_180, primals_181, buf196, buf202, buf204, 208, 64, grid=grid(208, 64), stream=stream0)
        del buf199
        del buf202
        # Topologically Sorted Source Nodes: [layer2_3_convs_2], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf206 = empty_strided_cuda((4, 208, 8, 8), (13312, 1, 1664, 208), torch.float32)
        # Topologically Sorted Source Nodes: [cat_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_34.run(buf203, buf205, primals_183, primals_184, primals_185, primals_186, buf196, buf206, 53248, grid=grid(53248), stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf208 = empty_strided_cuda((4, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_3_bn3, add_17, layer2_3_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_35.run(buf207, primals_188, primals_189, primals_190, primals_191, buf194, buf208, 131072, grid=grid(131072), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 416, 8, 8), (26624, 1, 3328, 416))
        buf210 = empty_strided_cuda((4, 416, 8, 8), (26624, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn1, layer3_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf209, primals_193, primals_194, primals_195, primals_196, buf210, 1664, 64, grid=grid(1664, 64), stream=stream0)
        buf211 = reinterpret_tensor(buf203, (4, 104, 8, 8), (6656, 1, 832, 104), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf210, buf211, 416, 64, grid=grid(416, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_convs_0], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, buf23, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf213 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_convs_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf210, buf213, 416, 64, grid=grid(416, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_convs_1], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, buf24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf215 = empty_strided_cuda((4, 208, 4, 4), (3328, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_22], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_39.run(buf212, primals_198, primals_199, primals_200, primals_201, buf214, primals_203, primals_204, primals_205, primals_206, buf215, 13312, grid=grid(13312), stream=stream0)
        buf216 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_convs_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_40.run(buf210, buf216, 416, 64, grid=grid(416, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_convs_2], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, buf25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf220 = reinterpret_tensor(buf216, (4, 416, 4, 4), (6656, 16, 4, 1), 0); del buf216  # reuse
        buf218 = reinterpret_tensor(buf220, (4, 104, 4, 4), (6656, 16, 4, 1), 4992)  # alias
        # Topologically Sorted Source Nodes: [layer3_0_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_41.run(buf210, buf218, 6656, grid=grid(6656), stream=stream0)
        buf219 = reinterpret_tensor(buf220, (4, 312, 4, 4), (6656, 16, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat_23], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_42.run(buf215, buf217, primals_208, primals_209, primals_210, primals_211, buf219, 19968, grid=grid(19968), stream=stream0)
        buf221 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_24], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_43.run(buf220, buf221, 1664, 16, grid=grid(1664, 16), stream=stream0)
        del buf218
        del buf219
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf208, primals_217, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf224 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        buf225 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_bn3, layer3_0_downsample_1, add_18, layer3_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf225, buf222, primals_213, primals_214, primals_215, primals_216, buf223, primals_218, primals_219, primals_220, primals_221, 65536, grid=grid(65536), stream=stream0)
        del primals_216
        del primals_221
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf227 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_bn1, layer3_1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf226, primals_223, primals_224, primals_225, primals_226, buf227, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf228 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_1_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf227, buf228, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_convs_0], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf234 = buf215; del buf215  # reuse
        buf230 = reinterpret_tensor(buf234, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf231 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_bns_0, layer3_1_relu_1, add_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf229, primals_228, primals_229, primals_230, primals_231, buf227, buf230, buf231, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_convs_1], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf233 = reinterpret_tensor(buf234, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf235 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_1_bns_1, layer3_1_relu_2, add_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf232, primals_233, primals_234, primals_235, primals_236, buf227, buf233, buf235, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf230
        del buf233
        # Topologically Sorted Source Nodes: [layer3_1_convs_2], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf237 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_27], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf234, buf236, primals_238, primals_239, primals_240, primals_241, buf227, buf237, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf239 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_1_bn3, add_21, layer3_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf238, primals_243, primals_244, primals_245, primals_246, buf225, buf239, 65536, grid=grid(65536), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf241 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_2_bn1, layer3_2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf240, primals_248, primals_249, primals_250, primals_251, buf241, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf242 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_2_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf241, buf242, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_convs_0], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf248 = buf234; del buf234  # reuse
        buf244 = reinterpret_tensor(buf248, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf245 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_bns_0, layer3_2_relu_1, add_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf243, primals_253, primals_254, primals_255, primals_256, buf241, buf244, buf245, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_convs_1], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf247 = reinterpret_tensor(buf248, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf249 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_2_bns_1, layer3_2_relu_2, add_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf246, primals_258, primals_259, primals_260, primals_261, buf241, buf247, buf249, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf244
        del buf247
        # Topologically Sorted Source Nodes: [layer3_2_convs_2], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf251 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_30], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf248, buf250, primals_263, primals_264, primals_265, primals_266, buf241, buf251, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf253 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_2_bn3, add_24, layer3_2_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf252, primals_268, primals_269, primals_270, primals_271, buf239, buf253, 65536, grid=grid(65536), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf255 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_3_bn1, layer3_3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf254, primals_273, primals_274, primals_275, primals_276, buf255, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf256 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_3_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf255, buf256, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_convs_0], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf262 = buf248; del buf248  # reuse
        buf258 = reinterpret_tensor(buf262, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf259 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_bns_0, layer3_3_relu_1, add_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf257, primals_278, primals_279, primals_280, primals_281, buf255, buf258, buf259, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_convs_1], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf261 = reinterpret_tensor(buf262, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf263 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_3_bns_1, layer3_3_relu_2, add_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf260, primals_283, primals_284, primals_285, primals_286, buf255, buf261, buf263, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf258
        del buf261
        # Topologically Sorted Source Nodes: [layer3_3_convs_2], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf265 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_33], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf262, buf264, primals_288, primals_289, primals_290, primals_291, buf255, buf265, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf267 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_3_bn3, add_27, layer3_3_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf266, primals_293, primals_294, primals_295, primals_296, buf253, buf267, 65536, grid=grid(65536), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf269 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_4_bn1, layer3_4_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf268, primals_298, primals_299, primals_300, primals_301, buf269, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf270 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_4_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf269, buf270, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_convs_0], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf276 = buf262; del buf262  # reuse
        buf272 = reinterpret_tensor(buf276, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf273 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_bns_0, layer3_4_relu_1, add_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf271, primals_303, primals_304, primals_305, primals_306, buf269, buf272, buf273, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_convs_1], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf275 = reinterpret_tensor(buf276, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf277 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_4_bns_1, layer3_4_relu_2, add_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf274, primals_308, primals_309, primals_310, primals_311, buf269, buf275, buf277, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf272
        del buf275
        # Topologically Sorted Source Nodes: [layer3_4_convs_2], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf279 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_36], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf276, buf278, primals_313, primals_314, primals_315, primals_316, buf269, buf279, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf281 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_4_bn3, add_30, layer3_4_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf280, primals_318, primals_319, primals_320, primals_321, buf267, buf281, 65536, grid=grid(65536), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf283 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_5_bn1, layer3_5_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf282, primals_323, primals_324, primals_325, primals_326, buf283, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf284 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_5_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf283, buf284, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_5_convs_0], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf290 = buf276; del buf276  # reuse
        buf286 = reinterpret_tensor(buf290, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf287 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_bns_0, layer3_5_relu_1, add_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf285, primals_328, primals_329, primals_330, primals_331, buf283, buf286, buf287, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_5_convs_1], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf289 = reinterpret_tensor(buf290, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf291 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_5_bns_1, layer3_5_relu_2, add_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf288, primals_333, primals_334, primals_335, primals_336, buf283, buf289, buf291, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf286
        del buf289
        # Topologically Sorted Source Nodes: [layer3_5_convs_2], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf293 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_39], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf290, buf292, primals_338, primals_339, primals_340, primals_341, buf283, buf293, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf295 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_5_bn3, add_33, layer3_5_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf294, primals_343, primals_344, primals_345, primals_346, buf281, buf295, 65536, grid=grid(65536), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [layer3_6_conv1], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf297 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_6_bn1, layer3_6_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf296, primals_348, primals_349, primals_350, primals_351, buf297, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf298 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_6_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf297, buf298, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_6_convs_0], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf304 = buf290; del buf290  # reuse
        buf300 = reinterpret_tensor(buf304, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf301 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [layer3_6_bns_0, layer3_6_relu_1, add_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf299, primals_353, primals_354, primals_355, primals_356, buf297, buf300, buf301, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_6_convs_1], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf303 = reinterpret_tensor(buf304, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf305 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_6_bns_1, layer3_6_relu_2, add_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf302, primals_358, primals_359, primals_360, primals_361, buf297, buf303, buf305, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf300
        del buf303
        # Topologically Sorted Source Nodes: [layer3_6_convs_2], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf307 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_42], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf304, buf306, primals_363, primals_364, primals_365, primals_366, buf297, buf307, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_6_conv3], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf309 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_6_bn3, add_36, layer3_6_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf308, primals_368, primals_369, primals_370, primals_371, buf295, buf309, 65536, grid=grid(65536), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [layer3_7_conv1], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf311 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_7_bn1, layer3_7_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf310, primals_373, primals_374, primals_375, primals_376, buf311, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf312 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_7_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf311, buf312, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_7_convs_0], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf318 = buf304; del buf304  # reuse
        buf314 = reinterpret_tensor(buf318, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf315 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [layer3_7_bns_0, layer3_7_relu_1, add_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf313, primals_378, primals_379, primals_380, primals_381, buf311, buf314, buf315, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_7_convs_1], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf317 = reinterpret_tensor(buf318, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf319 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_7_bns_1, layer3_7_relu_2, add_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf316, primals_383, primals_384, primals_385, primals_386, buf311, buf317, buf319, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf314
        del buf317
        # Topologically Sorted Source Nodes: [layer3_7_convs_2], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf321 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_45], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf318, buf320, primals_388, primals_389, primals_390, primals_391, buf311, buf321, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_7_conv3], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf323 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_7_bn3, add_39, layer3_7_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf322, primals_393, primals_394, primals_395, primals_396, buf309, buf323, 65536, grid=grid(65536), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [layer3_8_conv1], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf325 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_8_bn1, layer3_8_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf324, primals_398, primals_399, primals_400, primals_401, buf325, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf326 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_8_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf325, buf326, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_8_convs_0], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf332 = buf318; del buf318  # reuse
        buf328 = reinterpret_tensor(buf332, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf329 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [layer3_8_bns_0, layer3_8_relu_1, add_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf327, primals_403, primals_404, primals_405, primals_406, buf325, buf328, buf329, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_8_convs_1], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf331 = reinterpret_tensor(buf332, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf333 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_8_bns_1, layer3_8_relu_2, add_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf330, primals_408, primals_409, primals_410, primals_411, buf325, buf331, buf333, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf328
        del buf331
        # Topologically Sorted Source Nodes: [layer3_8_convs_2], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf335 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_48], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf332, buf334, primals_413, primals_414, primals_415, primals_416, buf325, buf335, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_8_conv3], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_417, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf337 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_8_bn3, add_42, layer3_8_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf336, primals_418, primals_419, primals_420, primals_421, buf323, buf337, 65536, grid=grid(65536), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [layer3_9_conv1], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf339 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_9_bn1, layer3_9_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf338, primals_423, primals_424, primals_425, primals_426, buf339, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf340 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_9_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf339, buf340, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_9_convs_0], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf346 = buf332; del buf332  # reuse
        buf342 = reinterpret_tensor(buf346, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf343 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [layer3_9_bns_0, layer3_9_relu_1, add_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf341, primals_428, primals_429, primals_430, primals_431, buf339, buf342, buf343, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_9_convs_1], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf345 = reinterpret_tensor(buf346, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf347 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_9_bns_1, layer3_9_relu_2, add_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf344, primals_433, primals_434, primals_435, primals_436, buf339, buf345, buf347, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf342
        del buf345
        # Topologically Sorted Source Nodes: [layer3_9_convs_2], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf349 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_51], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf346, buf348, primals_438, primals_439, primals_440, primals_441, buf339, buf349, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_9_conv3], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_442, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf351 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_9_bn3, add_45, layer3_9_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf350, primals_443, primals_444, primals_445, primals_446, buf337, buf351, 65536, grid=grid(65536), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [layer3_10_conv1], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf353 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_10_bn1, layer3_10_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf352, primals_448, primals_449, primals_450, primals_451, buf353, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf354 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_10_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf353, buf354, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_10_convs_0], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf360 = buf346; del buf346  # reuse
        buf356 = reinterpret_tensor(buf360, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf357 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [layer3_10_bns_0, layer3_10_relu_1, add_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf355, primals_453, primals_454, primals_455, primals_456, buf353, buf356, buf357, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_10_convs_1], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf359 = reinterpret_tensor(buf360, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf361 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_10_bns_1, layer3_10_relu_2, add_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf358, primals_458, primals_459, primals_460, primals_461, buf353, buf359, buf361, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf356
        del buf359
        # Topologically Sorted Source Nodes: [layer3_10_convs_2], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf363 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_54], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf360, buf362, primals_463, primals_464, primals_465, primals_466, buf353, buf363, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_10_conv3], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf365 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_10_bn3, add_48, layer3_10_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf364, primals_468, primals_469, primals_470, primals_471, buf351, buf365, 65536, grid=grid(65536), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [layer3_11_conv1], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf367 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_11_bn1, layer3_11_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf366, primals_473, primals_474, primals_475, primals_476, buf367, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf368 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_11_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf367, buf368, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_11_convs_0], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf374 = buf360; del buf360  # reuse
        buf370 = reinterpret_tensor(buf374, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf371 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [layer3_11_bns_0, layer3_11_relu_1, add_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf369, primals_478, primals_479, primals_480, primals_481, buf367, buf370, buf371, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_11_convs_1], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf373 = reinterpret_tensor(buf374, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf375 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_11_bns_1, layer3_11_relu_2, add_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf372, primals_483, primals_484, primals_485, primals_486, buf367, buf373, buf375, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf370
        del buf373
        # Topologically Sorted Source Nodes: [layer3_11_convs_2], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf377 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_57], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf374, buf376, primals_488, primals_489, primals_490, primals_491, buf367, buf377, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_11_conv3], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf379 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_11_bn3, add_51, layer3_11_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf378, primals_493, primals_494, primals_495, primals_496, buf365, buf379, 65536, grid=grid(65536), stream=stream0)
        del primals_496
        # Topologically Sorted Source Nodes: [layer3_12_conv1], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_497, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf381 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_12_bn1, layer3_12_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf380, primals_498, primals_499, primals_500, primals_501, buf381, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf382 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_12_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf381, buf382, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_12_convs_0], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf388 = buf374; del buf374  # reuse
        buf384 = reinterpret_tensor(buf388, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf385 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [layer3_12_bns_0, layer3_12_relu_1, add_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf383, primals_503, primals_504, primals_505, primals_506, buf381, buf384, buf385, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_12_convs_1], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, buf60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf387 = reinterpret_tensor(buf388, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf389 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_12_bns_1, layer3_12_relu_2, add_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf386, primals_508, primals_509, primals_510, primals_511, buf381, buf387, buf389, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf384
        del buf387
        # Topologically Sorted Source Nodes: [layer3_12_convs_2], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf391 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_60], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf388, buf390, primals_513, primals_514, primals_515, primals_516, buf381, buf391, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_12_conv3], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_517, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf393 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_12_bn3, add_54, layer3_12_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf392, primals_518, primals_519, primals_520, primals_521, buf379, buf393, 65536, grid=grid(65536), stream=stream0)
        del primals_521
        # Topologically Sorted Source Nodes: [layer3_13_conv1], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf395 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_13_bn1, layer3_13_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf394, primals_523, primals_524, primals_525, primals_526, buf395, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf396 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_13_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf395, buf396, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_13_convs_0], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, buf62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf402 = buf388; del buf388  # reuse
        buf398 = reinterpret_tensor(buf402, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf399 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [layer3_13_bns_0, layer3_13_relu_1, add_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf397, primals_528, primals_529, primals_530, primals_531, buf395, buf398, buf399, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_13_convs_1], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf401 = reinterpret_tensor(buf402, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf403 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_13_bns_1, layer3_13_relu_2, add_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf400, primals_533, primals_534, primals_535, primals_536, buf395, buf401, buf403, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf398
        del buf401
        # Topologically Sorted Source Nodes: [layer3_13_convs_2], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf405 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_63], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf402, buf404, primals_538, primals_539, primals_540, primals_541, buf395, buf405, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_13_conv3], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_542, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf407 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_13_bn3, add_57, layer3_13_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf406, primals_543, primals_544, primals_545, primals_546, buf393, buf407, 65536, grid=grid(65536), stream=stream0)
        del primals_546
        # Topologically Sorted Source Nodes: [layer3_14_conv1], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, primals_547, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf409 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_14_bn1, layer3_14_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf408, primals_548, primals_549, primals_550, primals_551, buf409, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf410 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_14_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf409, buf410, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_14_convs_0], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf416 = buf402; del buf402  # reuse
        buf412 = reinterpret_tensor(buf416, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf413 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [layer3_14_bns_0, layer3_14_relu_1, add_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf411, primals_553, primals_554, primals_555, primals_556, buf409, buf412, buf413, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_14_convs_1], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf415 = reinterpret_tensor(buf416, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf417 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_14_bns_1, layer3_14_relu_2, add_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf414, primals_558, primals_559, primals_560, primals_561, buf409, buf415, buf417, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf412
        del buf415
        # Topologically Sorted Source Nodes: [layer3_14_convs_2], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf419 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_66], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf416, buf418, primals_563, primals_564, primals_565, primals_566, buf409, buf419, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_14_conv3], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf419, primals_567, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf421 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_14_bn3, add_60, layer3_14_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf420, primals_568, primals_569, primals_570, primals_571, buf407, buf421, 65536, grid=grid(65536), stream=stream0)
        del primals_571
        # Topologically Sorted Source Nodes: [layer3_15_conv1], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_572, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf423 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_15_bn1, layer3_15_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf422, primals_573, primals_574, primals_575, primals_576, buf423, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf424 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_15_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf423, buf424, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_15_convs_0], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf430 = buf416; del buf416  # reuse
        buf426 = reinterpret_tensor(buf430, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf427 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [layer3_15_bns_0, layer3_15_relu_1, add_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf425, primals_578, primals_579, primals_580, primals_581, buf423, buf426, buf427, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_15_convs_1], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf429 = reinterpret_tensor(buf430, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf431 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_15_bns_1, layer3_15_relu_2, add_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf428, primals_583, primals_584, primals_585, primals_586, buf423, buf429, buf431, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf426
        del buf429
        # Topologically Sorted Source Nodes: [layer3_15_convs_2], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, buf70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf433 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_69], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf430, buf432, primals_588, primals_589, primals_590, primals_591, buf423, buf433, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_15_conv3], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_592, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf435 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_15_bn3, add_63, layer3_15_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf434, primals_593, primals_594, primals_595, primals_596, buf421, buf435, 65536, grid=grid(65536), stream=stream0)
        del primals_596
        # Topologically Sorted Source Nodes: [layer3_16_conv1], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, primals_597, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf437 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_16_bn1, layer3_16_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf436, primals_598, primals_599, primals_600, primals_601, buf437, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf438 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_16_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf437, buf438, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_16_convs_0], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf444 = buf430; del buf430  # reuse
        buf440 = reinterpret_tensor(buf444, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf441 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [layer3_16_bns_0, layer3_16_relu_1, add_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf439, primals_603, primals_604, primals_605, primals_606, buf437, buf440, buf441, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_16_convs_1], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf443 = reinterpret_tensor(buf444, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf445 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_16_bns_1, layer3_16_relu_2, add_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf442, primals_608, primals_609, primals_610, primals_611, buf437, buf443, buf445, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf440
        del buf443
        # Topologically Sorted Source Nodes: [layer3_16_convs_2], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf447 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_72], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf444, buf446, primals_613, primals_614, primals_615, primals_616, buf437, buf447, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_16_conv3], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_617, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf449 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_16_bn3, add_66, layer3_16_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf448, primals_618, primals_619, primals_620, primals_621, buf435, buf449, 65536, grid=grid(65536), stream=stream0)
        del primals_621
        # Topologically Sorted Source Nodes: [layer3_17_conv1], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, primals_622, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf451 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_17_bn1, layer3_17_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf450, primals_623, primals_624, primals_625, primals_626, buf451, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf452 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_17_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf451, buf452, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_17_convs_0], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, buf74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf458 = buf444; del buf444  # reuse
        buf454 = reinterpret_tensor(buf458, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf455 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [layer3_17_bns_0, layer3_17_relu_1, add_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf453, primals_628, primals_629, primals_630, primals_631, buf451, buf454, buf455, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_17_convs_1], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf457 = reinterpret_tensor(buf458, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf459 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_17_bns_1, layer3_17_relu_2, add_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf456, primals_633, primals_634, primals_635, primals_636, buf451, buf457, buf459, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf454
        del buf457
        # Topologically Sorted Source Nodes: [layer3_17_convs_2], Original ATen: [aten.convolution]
        buf460 = extern_kernels.convolution(buf459, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf461 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_75], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf458, buf460, primals_638, primals_639, primals_640, primals_641, buf451, buf461, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_17_conv3], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_642, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf463 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_17_bn3, add_69, layer3_17_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf462, primals_643, primals_644, primals_645, primals_646, buf449, buf463, 65536, grid=grid(65536), stream=stream0)
        del primals_646
        # Topologically Sorted Source Nodes: [layer3_18_conv1], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_647, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf465 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_18_bn1, layer3_18_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf464, primals_648, primals_649, primals_650, primals_651, buf465, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf466 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_18_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf465, buf466, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_18_convs_0], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf472 = buf458; del buf458  # reuse
        buf468 = reinterpret_tensor(buf472, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf469 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [layer3_18_bns_0, layer3_18_relu_1, add_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf467, primals_653, primals_654, primals_655, primals_656, buf465, buf468, buf469, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_18_convs_1], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf471 = reinterpret_tensor(buf472, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf473 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_18_bns_1, layer3_18_relu_2, add_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf470, primals_658, primals_659, primals_660, primals_661, buf465, buf471, buf473, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf468
        del buf471
        # Topologically Sorted Source Nodes: [layer3_18_convs_2], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf475 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_78], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf472, buf474, primals_663, primals_664, primals_665, primals_666, buf465, buf475, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_18_conv3], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_667, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf477 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_18_bn3, add_72, layer3_18_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf476, primals_668, primals_669, primals_670, primals_671, buf463, buf477, 65536, grid=grid(65536), stream=stream0)
        del primals_671
        # Topologically Sorted Source Nodes: [layer3_19_conv1], Original ATen: [aten.convolution]
        buf478 = extern_kernels.convolution(buf477, primals_672, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf478, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf479 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_19_bn1, layer3_19_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf478, primals_673, primals_674, primals_675, primals_676, buf479, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf480 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_19_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf479, buf480, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_19_convs_0], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf486 = buf472; del buf472  # reuse
        buf482 = reinterpret_tensor(buf486, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf483 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [layer3_19_bns_0, layer3_19_relu_1, add_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf481, primals_678, primals_679, primals_680, primals_681, buf479, buf482, buf483, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_19_convs_1], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, buf81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf485 = reinterpret_tensor(buf486, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf487 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_19_bns_1, layer3_19_relu_2, add_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf484, primals_683, primals_684, primals_685, primals_686, buf479, buf485, buf487, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf482
        del buf485
        # Topologically Sorted Source Nodes: [layer3_19_convs_2], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf487, buf82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf489 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_81], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf486, buf488, primals_688, primals_689, primals_690, primals_691, buf479, buf489, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_19_conv3], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, primals_692, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf491 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_19_bn3, add_75, layer3_19_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf490, primals_693, primals_694, primals_695, primals_696, buf477, buf491, 65536, grid=grid(65536), stream=stream0)
        del primals_696
        # Topologically Sorted Source Nodes: [layer3_20_conv1], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf491, primals_697, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf493 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_20_bn1, layer3_20_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf492, primals_698, primals_699, primals_700, primals_701, buf493, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf494 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_20_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf493, buf494, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_20_convs_0], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf500 = buf486; del buf486  # reuse
        buf496 = reinterpret_tensor(buf500, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf497 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [layer3_20_bns_0, layer3_20_relu_1, add_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf495, primals_703, primals_704, primals_705, primals_706, buf493, buf496, buf497, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_20_convs_1], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, buf84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf499 = reinterpret_tensor(buf500, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf501 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_20_bns_1, layer3_20_relu_2, add_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf498, primals_708, primals_709, primals_710, primals_711, buf493, buf499, buf501, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf496
        del buf499
        # Topologically Sorted Source Nodes: [layer3_20_convs_2], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, buf85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf503 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_84], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf500, buf502, primals_713, primals_714, primals_715, primals_716, buf493, buf503, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_20_conv3], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, primals_717, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf505 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_20_bn3, add_78, layer3_20_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf504, primals_718, primals_719, primals_720, primals_721, buf491, buf505, 65536, grid=grid(65536), stream=stream0)
        del primals_721
        # Topologically Sorted Source Nodes: [layer3_21_conv1], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_722, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf507 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_21_bn1, layer3_21_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf506, primals_723, primals_724, primals_725, primals_726, buf507, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf508 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_21_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf507, buf508, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_21_convs_0], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf514 = buf500; del buf500  # reuse
        buf510 = reinterpret_tensor(buf514, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf511 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [layer3_21_bns_0, layer3_21_relu_1, add_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf509, primals_728, primals_729, primals_730, primals_731, buf507, buf510, buf511, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_21_convs_1], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf513 = reinterpret_tensor(buf514, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf515 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_21_bns_1, layer3_21_relu_2, add_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf512, primals_733, primals_734, primals_735, primals_736, buf507, buf513, buf515, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf510
        del buf513
        # Topologically Sorted Source Nodes: [layer3_21_convs_2], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf517 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_87], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf514, buf516, primals_738, primals_739, primals_740, primals_741, buf507, buf517, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_21_conv3], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, primals_742, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf519 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_21_bn3, add_81, layer3_21_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf518, primals_743, primals_744, primals_745, primals_746, buf505, buf519, 65536, grid=grid(65536), stream=stream0)
        del primals_746
        # Topologically Sorted Source Nodes: [layer3_22_conv1], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, primals_747, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 416, 4, 4), (6656, 1, 1664, 416))
        buf521 = empty_strided_cuda((4, 416, 4, 4), (6656, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_22_bn1, layer3_22_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_45.run(buf520, primals_748, primals_749, primals_750, primals_751, buf521, 1664, 16, grid=grid(1664, 16), stream=stream0)
        buf522 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_22_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf521, buf522, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_22_convs_0], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf528 = buf514; del buf514  # reuse
        buf524 = reinterpret_tensor(buf528, (4, 104, 4, 4), (3328, 16, 4, 1), 0)  # alias
        buf525 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [layer3_22_bns_0, layer3_22_relu_1, add_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf523, primals_753, primals_754, primals_755, primals_756, buf521, buf524, buf525, 416, 16, grid=grid(416, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_22_convs_1], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf527 = reinterpret_tensor(buf528, (4, 104, 4, 4), (3328, 16, 4, 1), 1664)  # alias
        buf529 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_22_bns_1, layer3_22_relu_2, add_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf526, primals_758, primals_759, primals_760, primals_761, buf521, buf527, buf529, 416, 16, grid=grid(416, 16), stream=stream0)
        del buf524
        del buf527
        # Topologically Sorted Source Nodes: [layer3_22_convs_2], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf529, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf531 = empty_strided_cuda((4, 416, 4, 4), (6656, 1, 1664, 416), torch.float32)
        # Topologically Sorted Source Nodes: [cat_90], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_49.run(buf528, buf530, primals_763, primals_764, primals_765, primals_766, buf521, buf531, 26624, grid=grid(26624), stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_22_conv3], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf531, primals_767, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf533 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_22_bn3, add_84, layer3_22_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf532, primals_768, primals_769, primals_770, primals_771, buf519, buf533, 65536, grid=grid(65536), stream=stream0)
        del primals_771
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_772, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 832, 4, 4), (13312, 1, 3328, 832))
        buf535 = empty_strided_cuda((4, 832, 4, 4), (13312, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn1, layer4_0_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf534, primals_773, primals_774, primals_775, primals_776, buf535, 3328, 16, grid=grid(3328, 16), stream=stream0)
        buf536 = reinterpret_tensor(buf528, (4, 208, 4, 4), (3328, 1, 832, 208), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(buf535, buf536, 832, 16, grid=grid(832, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_convs_0], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf536, buf92, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf537, (4, 208, 2, 2), (832, 1, 416, 208))
        buf538 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_convs_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_53.run(buf535, buf538, 832, 16, grid=grid(832, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_convs_1], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, buf93, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (4, 208, 2, 2), (832, 1, 416, 208))
        buf540 = empty_strided_cuda((4, 416, 2, 2), (1664, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_91], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_54.run(buf537, primals_778, primals_779, primals_780, primals_781, buf539, primals_783, primals_784, primals_785, primals_786, buf540, 6656, grid=grid(6656), stream=stream0)
        buf541 = buf538; del buf538  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_convs_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf535, buf541, 832, 16, grid=grid(832, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_convs_2], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, buf94, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (4, 208, 2, 2), (832, 1, 416, 208))
        buf545 = reinterpret_tensor(buf541, (4, 832, 2, 2), (3328, 4, 2, 1), 0); del buf541  # reuse
        buf543 = reinterpret_tensor(buf545, (4, 208, 2, 2), (3328, 4, 2, 1), 2496)  # alias
        # Topologically Sorted Source Nodes: [layer4_0_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_56.run(buf535, buf543, 3328, grid=grid(3328), stream=stream0)
        buf544 = reinterpret_tensor(buf545, (4, 624, 2, 2), (3328, 4, 2, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat_92], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_57.run(buf540, buf542, primals_788, primals_789, primals_790, primals_791, buf544, 9984, grid=grid(9984), stream=stream0)
        buf546 = empty_strided_cuda((4, 832, 2, 2), (3328, 1, 1664, 832), torch.float32)
        # Topologically Sorted Source Nodes: [cat_93], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_58.run(buf545, buf546, 3328, 4, grid=grid(3328, 4), stream=stream0)
        del buf543
        del buf544
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, primals_792, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf533, primals_797, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf549 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        buf550 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_bn3, layer4_0_downsample_1, add_85, layer4_0_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_59.run(buf550, buf547, primals_793, primals_794, primals_795, primals_796, buf548, primals_798, primals_799, primals_800, primals_801, 32768, grid=grid(32768), stream=stream0)
        del primals_796
        del primals_801
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_802, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 832, 2, 2), (3328, 1, 1664, 832))
        buf552 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_bn1, layer4_1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf551, primals_803, primals_804, primals_805, primals_806, buf552, 3328, 4, grid=grid(3328, 4), stream=stream0)
        buf553 = empty_strided_cuda((4, 208, 2, 2), (832, 1, 416, 208), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_1_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_61.run(buf552, buf553, 832, 4, grid=grid(832, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_convs_0], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, buf95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (4, 208, 2, 2), (832, 1, 416, 208))
        buf559 = buf540; del buf540  # reuse
        buf555 = reinterpret_tensor(buf559, (4, 208, 2, 2), (1664, 4, 2, 1), 0)  # alias
        buf556 = buf553; del buf553  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_bns_0, layer4_1_relu_1, add_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_62.run(buf554, primals_808, primals_809, primals_810, primals_811, buf552, buf555, buf556, 832, 4, grid=grid(832, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_convs_1], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (4, 208, 2, 2), (832, 1, 416, 208))
        buf558 = reinterpret_tensor(buf559, (4, 208, 2, 2), (1664, 4, 2, 1), 832)  # alias
        buf560 = empty_strided_cuda((4, 208, 2, 2), (832, 1, 416, 208), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_1_bns_1, layer4_1_relu_2, add_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf557, primals_813, primals_814, primals_815, primals_816, buf552, buf558, buf560, 832, 4, grid=grid(832, 4), stream=stream0)
        del buf555
        del buf558
        # Topologically Sorted Source Nodes: [layer4_1_convs_2], Original ATen: [aten.convolution]
        buf561 = extern_kernels.convolution(buf560, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf561, (4, 208, 2, 2), (832, 1, 416, 208))
        buf562 = empty_strided_cuda((4, 832, 2, 2), (3328, 1, 1664, 832), torch.float32)
        # Topologically Sorted Source Nodes: [cat_96], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf559, buf561, primals_818, primals_819, primals_820, primals_821, buf552, buf562, 13312, grid=grid(13312), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, primals_822, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf564 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_1_bn3, add_88, layer4_1_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_65.run(buf563, primals_823, primals_824, primals_825, primals_826, buf550, buf564, 32768, grid=grid(32768), stream=stream0)
        del primals_826
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten.convolution]
        buf565 = extern_kernels.convolution(buf564, primals_827, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf565, (4, 832, 2, 2), (3328, 1, 1664, 832))
        buf566 = empty_strided_cuda((4, 832, 2, 2), (3328, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_2_bn1, layer4_2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf565, primals_828, primals_829, primals_830, primals_831, buf566, 3328, 4, grid=grid(3328, 4), stream=stream0)
        buf567 = empty_strided_cuda((4, 208, 2, 2), (832, 1, 416, 208), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_2_convs_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_61.run(buf566, buf567, 832, 4, grid=grid(832, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_2_convs_0], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf567, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf568, (4, 208, 2, 2), (832, 1, 416, 208))
        buf573 = buf559; del buf559  # reuse
        buf569 = reinterpret_tensor(buf573, (4, 208, 2, 2), (1664, 4, 2, 1), 0)  # alias
        buf570 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_bns_0, layer4_2_relu_1, add_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_62.run(buf568, primals_833, primals_834, primals_835, primals_836, buf566, buf569, buf570, 832, 4, grid=grid(832, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_2_convs_1], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, buf99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (4, 208, 2, 2), (832, 1, 416, 208))
        buf572 = reinterpret_tensor(buf573, (4, 208, 2, 2), (1664, 4, 2, 1), 832)  # alias
        buf574 = empty_strided_cuda((4, 208, 2, 2), (832, 1, 416, 208), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_2_bns_1, layer4_2_relu_2, add_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf571, primals_838, primals_839, primals_840, primals_841, buf566, buf572, buf574, 832, 4, grid=grid(832, 4), stream=stream0)
        del buf569
        del buf572
        # Topologically Sorted Source Nodes: [layer4_2_convs_2], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf574, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf575, (4, 208, 2, 2), (832, 1, 416, 208))
        buf576 = empty_strided_cuda((4, 832, 2, 2), (3328, 1, 1664, 832), torch.float32)
        # Topologically Sorted Source Nodes: [cat_99], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_64.run(buf573, buf575, primals_843, primals_844, primals_845, primals_846, buf566, buf576, 13312, grid=grid(13312), stream=stream0)
        del buf573
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(buf576, primals_847, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf577, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf578 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float32)
        buf581 = empty_strided_cuda((4, 2048, 2, 2), (8192, 1, 4096, 2048), torch.bool)
        # Topologically Sorted Source Nodes: [layer4_2_bn3, add_91, layer4_2_relu_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_66.run(buf577, primals_848, primals_849, primals_850, primals_851, buf564, buf578, buf581, 32768, grid=grid(32768), stream=stream0)
        del primals_851
        buf579 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [avgpool], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_67.run(buf578, buf579, 8192, grid=grid(8192), stream=stream0)
        del buf578
        buf580 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_853, reinterpret_tensor(buf579, (4, 2048), (2048, 1), 0), reinterpret_tensor(primals_852, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf580)
        del primals_853
    return (buf580, buf0, buf1, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, buf2, primals_13, primals_14, primals_15, primals_16, buf3, primals_18, primals_19, primals_20, primals_21, buf4, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_41, buf5, primals_43, primals_44, primals_45, primals_46, buf6, primals_48, primals_49, primals_50, primals_51, buf7, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, buf8, primals_68, primals_69, primals_70, primals_71, buf9, primals_73, primals_74, primals_75, primals_76, buf10, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, buf11, primals_93, primals_94, primals_95, primals_96, buf12, primals_98, primals_99, primals_100, primals_101, buf13, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, buf14, primals_123, primals_124, primals_125, primals_126, buf15, primals_128, primals_129, primals_130, primals_131, buf16, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_146, buf17, primals_148, primals_149, primals_150, primals_151, buf18, primals_153, primals_154, primals_155, primals_156, buf19, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, buf20, primals_173, primals_174, primals_175, primals_176, buf21, primals_178, primals_179, primals_180, primals_181, buf22, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_196, buf23, primals_198, primals_199, primals_200, primals_201, buf24, primals_203, primals_204, primals_205, primals_206, buf25, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_226, buf26, primals_228, primals_229, primals_230, primals_231, buf27, primals_233, primals_234, primals_235, primals_236, buf28, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_251, buf29, primals_253, primals_254, primals_255, primals_256, buf30, primals_258, primals_259, primals_260, primals_261, buf31, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_276, buf32, primals_278, primals_279, primals_280, primals_281, buf33, primals_283, primals_284, primals_285, primals_286, buf34, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_301, buf35, primals_303, primals_304, primals_305, primals_306, buf36, primals_308, primals_309, primals_310, primals_311, buf37, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_326, buf38, primals_328, primals_329, primals_330, primals_331, buf39, primals_333, primals_334, primals_335, primals_336, buf40, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_347, primals_348, primals_349, primals_350, primals_351, buf41, primals_353, primals_354, primals_355, primals_356, buf42, primals_358, primals_359, primals_360, primals_361, buf43, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_376, buf44, primals_378, primals_379, primals_380, primals_381, buf45, primals_383, primals_384, primals_385, primals_386, buf46, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_401, buf47, primals_403, primals_404, primals_405, primals_406, buf48, primals_408, primals_409, primals_410, primals_411, buf49, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_426, buf50, primals_428, primals_429, primals_430, primals_431, buf51, primals_433, primals_434, primals_435, primals_436, buf52, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, primals_451, buf53, primals_453, primals_454, primals_455, primals_456, buf54, primals_458, primals_459, primals_460, primals_461, buf55, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_475, primals_476, buf56, primals_478, primals_479, primals_480, primals_481, buf57, primals_483, primals_484, primals_485, primals_486, buf58, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_497, primals_498, primals_499, primals_500, primals_501, buf59, primals_503, primals_504, primals_505, primals_506, buf60, primals_508, primals_509, primals_510, primals_511, buf61, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_522, primals_523, primals_524, primals_525, primals_526, buf62, primals_528, primals_529, primals_530, primals_531, buf63, primals_533, primals_534, primals_535, primals_536, buf64, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_547, primals_548, primals_549, primals_550, primals_551, buf65, primals_553, primals_554, primals_555, primals_556, buf66, primals_558, primals_559, primals_560, primals_561, buf67, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_572, primals_573, primals_574, primals_575, primals_576, buf68, primals_578, primals_579, primals_580, primals_581, buf69, primals_583, primals_584, primals_585, primals_586, buf70, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_597, primals_598, primals_599, primals_600, primals_601, buf71, primals_603, primals_604, primals_605, primals_606, buf72, primals_608, primals_609, primals_610, primals_611, buf73, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_622, primals_623, primals_624, primals_625, primals_626, buf74, primals_628, primals_629, primals_630, primals_631, buf75, primals_633, primals_634, primals_635, primals_636, buf76, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_647, primals_648, primals_649, primals_650, primals_651, buf77, primals_653, primals_654, primals_655, primals_656, buf78, primals_658, primals_659, primals_660, primals_661, buf79, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_672, primals_673, primals_674, primals_675, primals_676, buf80, primals_678, primals_679, primals_680, primals_681, buf81, primals_683, primals_684, primals_685, primals_686, buf82, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_697, primals_698, primals_699, primals_700, primals_701, buf83, primals_703, primals_704, primals_705, primals_706, buf84, primals_708, primals_709, primals_710, primals_711, buf85, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_722, primals_723, primals_724, primals_725, primals_726, buf86, primals_728, primals_729, primals_730, primals_731, buf87, primals_733, primals_734, primals_735, primals_736, buf88, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_747, primals_748, primals_749, primals_750, primals_751, buf89, primals_753, primals_754, primals_755, primals_756, buf90, primals_758, primals_759, primals_760, primals_761, buf91, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_772, primals_773, primals_774, primals_775, primals_776, buf92, primals_778, primals_779, primals_780, primals_781, buf93, primals_783, primals_784, primals_785, primals_786, buf94, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_797, primals_798, primals_799, primals_800, primals_802, primals_803, primals_804, primals_805, primals_806, buf95, primals_808, primals_809, primals_810, primals_811, buf96, primals_813, primals_814, primals_815, primals_816, buf97, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_827, primals_828, primals_829, primals_830, primals_831, buf98, primals_833, primals_834, primals_835, primals_836, buf99, primals_838, primals_839, primals_840, primals_841, buf100, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, buf101, buf102, buf103, buf104, buf105, reinterpret_tensor(buf106, (4, 26, 16, 16), (26624, 256, 16, 1), 0), buf108, reinterpret_tensor(buf106, (4, 26, 16, 16), (26624, 256, 16, 1), 6656), buf110, reinterpret_tensor(buf106, (4, 26, 16, 16), (26624, 256, 16, 1), 13312), buf113, reinterpret_tensor(buf106, (4, 26, 16, 16), (26624, 256, 16, 1), 19968), buf117, buf118, buf119, buf121, buf122, reinterpret_tensor(buf123, (4, 26, 16, 16), (26624, 256, 16, 1), 0), buf125, buf127, buf128, buf131, buf132, buf133, buf134, buf135, buf136, reinterpret_tensor(buf137, (4, 26, 16, 16), (26624, 256, 16, 1), 0), buf139, buf141, buf142, buf145, buf146, buf147, buf148, buf149, buf150, reinterpret_tensor(buf151, (4, 52, 16, 16), (53248, 256, 16, 1), 0), buf153, reinterpret_tensor(buf151, (4, 52, 16, 16), (53248, 256, 16, 1), 13312), buf155, reinterpret_tensor(buf151, (4, 52, 16, 16), (53248, 256, 16, 1), 26624), buf158, reinterpret_tensor(buf151, (4, 52, 16, 16), (53248, 256, 16, 1), 39936), buf162, buf163, buf164, buf166, buf167, reinterpret_tensor(buf168, (4, 52, 8, 8), (13312, 64, 8, 1), 0), buf170, buf172, buf173, buf176, buf177, buf178, buf179, buf180, buf181, reinterpret_tensor(buf182, (4, 52, 8, 8), (13312, 64, 8, 1), 0), buf184, buf186, buf187, buf190, buf191, buf192, buf193, buf194, buf195, reinterpret_tensor(buf196, (4, 52, 8, 8), (13312, 64, 8, 1), 0), buf198, buf200, buf201, buf204, buf205, buf206, buf207, buf208, buf209, reinterpret_tensor(buf210, (4, 104, 8, 8), (26624, 64, 8, 1), 0), buf212, reinterpret_tensor(buf210, (4, 104, 8, 8), (26624, 64, 8, 1), 6656), buf214, reinterpret_tensor(buf210, (4, 104, 8, 8), (26624, 64, 8, 1), 13312), buf217, reinterpret_tensor(buf210, (4, 104, 8, 8), (26624, 64, 8, 1), 19968), buf221, buf222, buf223, buf225, buf226, reinterpret_tensor(buf227, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf229, buf231, buf232, buf235, buf236, buf237, buf238, buf239, buf240, reinterpret_tensor(buf241, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf243, buf245, buf246, buf249, buf250, buf251, buf252, buf253, buf254, reinterpret_tensor(buf255, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf257, buf259, buf260, buf263, buf264, buf265, buf266, buf267, buf268, reinterpret_tensor(buf269, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf271, buf273, buf274, buf277, buf278, buf279, buf280, buf281, buf282, reinterpret_tensor(buf283, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf285, buf287, buf288, buf291, buf292, buf293, buf294, buf295, buf296, reinterpret_tensor(buf297, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf299, buf301, buf302, buf305, buf306, buf307, buf308, buf309, buf310, reinterpret_tensor(buf311, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf313, buf315, buf316, buf319, buf320, buf321, buf322, buf323, buf324, reinterpret_tensor(buf325, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf327, buf329, buf330, buf333, buf334, buf335, buf336, buf337, buf338, reinterpret_tensor(buf339, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf341, buf343, buf344, buf347, buf348, buf349, buf350, buf351, buf352, reinterpret_tensor(buf353, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf355, buf357, buf358, buf361, buf362, buf363, buf364, buf365, buf366, reinterpret_tensor(buf367, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf369, buf371, buf372, buf375, buf376, buf377, buf378, buf379, buf380, reinterpret_tensor(buf381, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf383, buf385, buf386, buf389, buf390, buf391, buf392, buf393, buf394, reinterpret_tensor(buf395, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf397, buf399, buf400, buf403, buf404, buf405, buf406, buf407, buf408, reinterpret_tensor(buf409, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf411, buf413, buf414, buf417, buf418, buf419, buf420, buf421, buf422, reinterpret_tensor(buf423, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf425, buf427, buf428, buf431, buf432, buf433, buf434, buf435, buf436, reinterpret_tensor(buf437, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf439, buf441, buf442, buf445, buf446, buf447, buf448, buf449, buf450, reinterpret_tensor(buf451, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf453, buf455, buf456, buf459, buf460, buf461, buf462, buf463, buf464, reinterpret_tensor(buf465, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf467, buf469, buf470, buf473, buf474, buf475, buf476, buf477, buf478, reinterpret_tensor(buf479, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf481, buf483, buf484, buf487, buf488, buf489, buf490, buf491, buf492, reinterpret_tensor(buf493, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf495, buf497, buf498, buf501, buf502, buf503, buf504, buf505, buf506, reinterpret_tensor(buf507, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf509, buf511, buf512, buf515, buf516, buf517, buf518, buf519, buf520, reinterpret_tensor(buf521, (4, 104, 4, 4), (6656, 16, 4, 1), 0), buf523, buf525, buf526, buf529, buf530, buf531, buf532, buf533, buf534, reinterpret_tensor(buf535, (4, 208, 4, 4), (13312, 16, 4, 1), 0), buf537, reinterpret_tensor(buf535, (4, 208, 4, 4), (13312, 16, 4, 1), 3328), buf539, reinterpret_tensor(buf535, (4, 208, 4, 4), (13312, 16, 4, 1), 6656), buf542, reinterpret_tensor(buf535, (4, 208, 4, 4), (13312, 16, 4, 1), 9984), buf546, buf547, buf548, buf550, buf551, reinterpret_tensor(buf552, (4, 208, 2, 2), (3328, 4, 2, 1), 0), buf554, buf556, buf557, buf560, buf561, buf562, buf563, buf564, buf565, reinterpret_tensor(buf566, (4, 208, 2, 2), (3328, 4, 2, 1), 0), buf568, buf570, buf571, buf574, buf575, buf576, buf577, reinterpret_tensor(buf579, (4, 2048), (2048, 1), 0), primals_852, buf581, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((104, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((208, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((416, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((832, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
